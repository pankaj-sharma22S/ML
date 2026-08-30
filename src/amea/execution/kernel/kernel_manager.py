"""Kernel Manager handling multi-tenant Jupyter client lifecycle, sessions, and IOPub communication."""

import queue
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import psutil
from jupyter_client import KernelManager as JupyterKernelManager

from amea.execution.kernel.kernel_config import KernelConfig
from amea.execution.kernel.kernel_session import KernelSession, KernelStatus
from amea.execution.kernel.output_parser import OutputParser
from amea.execution.kernel.execution_result import CellOutput
from amea.execution.security import EnvironmentSanitizer


class SessionHandle:
    """Internal wrapper holding the live Jupyter KernelManager and client connection."""
    def __init__(self, session: KernelSession, km: JupyterKernelManager, client: Any, workspace_dir: Path):
        self.session = session
        self.km = km
        self.client = client
        self.workspace_dir = workspace_dir
        self.artifact_dir = workspace_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


class KernelManager:
    """Manages project-isolated interactive Python kernels with lifecycle controls."""

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self._sessions: Dict[str, SessionHandle] = {}

    def create_session(
        self,
        project_id: str,
        session_id: Optional[str] = None,
    ) -> KernelSession:
        """Spawn a new isolated Jupyter Python kernel for a project."""
        sid = session_id or f"sess_{uuid4().hex[:8]}"
        if sid in self._sessions:
            return self._sessions[sid].session

        workspace_dir = Path(".").resolve()
        artifact_dir = workspace_dir / ".amea_project" / "sessions" / sid / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        clean_env = EnvironmentSanitizer.sanitize()

        km = JupyterKernelManager(kernel_name=self.config.kernel_name)
        km.start_kernel(cwd=str(workspace_dir), env=clean_env)
        client = km.client()
        client.start_channels()

        # Wait for kernel to be ready and initialize matplotlib inline
        try:
            client.wait_for_ready(timeout=15)
            client.execute("%matplotlib inline")
            time.sleep(0.3)
            while True:
                try:
                    client.get_iopub_msg(timeout=0.2)
                except queue.Empty:
                    break
        except Exception:
            pass

        session = KernelSession(
            session_id=sid,
            project_id=project_id,
            kernel_id=getattr(km, "kernel_id", sid),
            status=KernelStatus.IDLE,
            workspace_path=str(workspace_dir.resolve()),
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
        )

        handle = SessionHandle(session, km, client, workspace_dir)
        self._sessions[sid] = handle
        return session

    def get_session(self, session_id: str) -> Optional[KernelSession]:
        """Retrieve kernel session state with updated resource usage."""
        handle = self._sessions.get(session_id)
        if not handle:
            return None

        # Check if process is still alive and update metrics
        if not self.is_alive(session_id):
            handle.session.status = KernelStatus.TERMINATED
        else:
            self._update_resource_usage(handle)

        return handle.session

    def execute(
        self,
        session_id: str,
        code: str,
        timeout_seconds: Optional[int] = None,
    ) -> Tuple[List[CellOutput], int]:
        """
        Execute code string in the specified session's kernel.
        Returns (list_of_outputs, execution_count).
        """
        handle = self._sessions.get(session_id)
        if not handle:
            raise ValueError(f"Session '{session_id}' does not exist or has been terminated.")

        timeout = timeout_seconds or self.config.default_timeout_seconds
        handle.session.status = KernelStatus.BUSY
        handle.session.last_activity = datetime.now(timezone.utc)
        client = handle.client

        msg_id = client.execute(code)
        outputs: List[CellOutput] = []
        execution_count = handle.session.execution_count + 1

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.interrupt(session_id)
                handle.session.status = KernelStatus.IDLE
                raise TimeoutError(f"Cell execution exceeded timeout of {timeout}s.")

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                if not self.is_alive(session_id):
                    handle.session.status = KernelStatus.ERROR
                    raise RuntimeError("Kernel process died during execution.")
                continue

            # Verify message belongs to our execution request
            parent_id = msg.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue

            msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})

            if msg_type == "status":
                exec_state = content.get("execution_state")
                if exec_state == "idle":
                    break

            if msg_type in ("stream", "display_data", "execute_result", "error"):
                if msg_type == "execute_result":
                    execution_count = content.get("execution_count", execution_count)

                parsed_output = OutputParser.parse_iopub_message(msg, handle.artifact_dir)
                if parsed_output:
                    outputs.append(parsed_output)

        handle.session.execution_count = execution_count
        handle.session.status = KernelStatus.IDLE
        handle.session.last_activity = datetime.now(timezone.utc)

        return outputs, execution_count

    def interrupt(self, session_id: str) -> bool:
        """Interrupt currently running execution in kernel."""
        handle = self._sessions.get(session_id)
        if not handle or not self.is_alive(session_id):
            return False

        handle.session.status = KernelStatus.INTERRUPTING
        try:
            handle.km.interrupt_kernel()
            handle.session.status = KernelStatus.IDLE
            return True
        except Exception:
            return False

    def restart(self, session_id: str) -> bool:
        """Restart kernel, resetting all in-memory variables and state."""
        handle = self._sessions.get(session_id)
        if not handle:
            return False

        handle.session.status = KernelStatus.RESTARTING
        try:
            handle.km.restart_kernel(now=True)
            handle.client.wait_for_ready(timeout=15)
            handle.session.status = KernelStatus.IDLE
            handle.session.execution_count = 0
            handle.session.last_activity = datetime.now(timezone.utc)
            return True
        except Exception:
            handle.session.status = KernelStatus.ERROR
            return False

    def shutdown(self, session_id: str) -> bool:
        """Terminate the kernel process and clean up channels."""
        handle = self._sessions.pop(session_id, None)
        if not handle:
            return False

        try:
            handle.client.stop_channels()
            handle.km.shutdown_kernel(now=True)
            handle.session.status = KernelStatus.TERMINATED
            return True
        except Exception:
            return False

    def is_alive(self, session_id: str) -> bool:
        """Check if kernel process is active."""
        handle = self._sessions.get(session_id)
        if not handle:
            return False
        return handle.km.is_alive()

    def list_sessions(self, project_id: Optional[str] = None) -> List[KernelSession]:
        """List active sessions, optionally filtered by project_id."""
        sessions = [h.session for h in self._sessions.values()]
        if project_id:
            return [s for s in sessions if s.project_id == project_id]
        return sessions

    def _update_resource_usage(self, handle: SessionHandle) -> None:
        """Sample CPU and memory usage of the kernel process."""
        try:
            pid = getattr(handle.km, "kernel", None)
            proc_pid = pid.pid if pid else None
            if proc_pid and psutil.pid_exists(proc_pid):
                p = psutil.Process(proc_pid)
                handle.session.cpu_usage_percent = p.cpu_percent(interval=None)
                handle.session.memory_usage_mb = round(p.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass
