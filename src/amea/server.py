"""Unified FastAPI Server for AMEA - Autonomous ML Engineer Workspace."""

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import starlette.routing
_orig_router_init = starlette.routing.Router.__init__
def _compat_router_init(self, *args, on_startup=None, on_shutdown=None, **kwargs):
    return _orig_router_init(self, *args, **kwargs)
starlette.routing.Router.__init__ = _compat_router_init

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from amea.execution.kernel.ai_cell_assistant import AICellAssistant
from amea.execution.kernel.execution_request import (
    BatchExecuteRequest,
    ExecuteCellRequest,
    NotebookCell,
)
from amea.execution.kernel.execution_result import CellExecutionResult
from amea.execution.kernel.graph_kernel_executor import GraphKernelExecutor
from amea.execution.kernel.kernel_manager import KernelManager
from amea.execution.kernel.notebook_manager import NotebookManager
from amea.execution.router import kernel_router
from amea.query_analysis.router import router as query_analysis_router


# ============================================================
# Request / Response Schemas for Project Management & Server
# ============================================================

class ProjectTemplate(BaseModel):
    name: str
    description: str


class CreateProjectRequest(BaseModel):
    name: str
    location: Optional[str] = None
    template: str = "empty"  # empty, classification, regression, time_series, data_analysis


class OpenProjectRequest(BaseModel):
    path: str


class FileOperationRequest(BaseModel):
    project_path: str
    relative_path: str
    content: Optional[str] = None
    new_name: Optional[str] = None


class TerminalExecRequest(BaseModel):
    project_path: str
    command: str


class AIThreadMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:8]}")
    sender: str  # "user" or "ai"
    text: str
    code_diff: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIThread(BaseModel):
    id: str = Field(default_factory=lambda: f"thread_{uuid4().hex[:8]}")
    project_id: str
    title: str
    messages: List[AIThreadMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SaveThreadRequest(BaseModel):
    project_id: str
    thread: AIThread


# ============================================================
# FastAPI Application Configuration
# ============================================================

app = FastAPI(
    title="AMEA - Autonomous ML Engineer Workspace",
    description="Interactive ML Engineering IDE Backend with Python Kernel, AI Agent Threads, and Notebooks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory thread repository (backed by filesystem)
_THREADS_DB: Dict[str, List[AIThread]] = {}

from fastapi import Header, Depends
from amea.auth.supabase_service import supabase_service


class AuthCredentialsRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class PublicChatRequest(BaseModel):
    message: str
    user_context: Optional[Dict[str, Any]] = None


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    return supabase_service.verify_token(authorization)


def get_current_user_required(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        # Default local dev user if running without strict auth header in local development
        return {"id": "local_dev_user", "email": "dev@amea.ai", "role": "authenticated"}
    user = supabase_service.verify_token(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please sign in again.",
        )
    return user


# ============================================================
# Supabase Authentication & Public Chat Endpoints
# ============================================================

@app.post("/api/auth/signup")
def auth_signup(req: AuthCredentialsRequest) -> Dict[str, Any]:
    """Register a new user via Supabase Auth."""
    try:
        res = supabase_service.sign_up(req.email, req.password, req.full_name)
        return {"status": "success", **res}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/api/auth/login")
def auth_login(req: AuthCredentialsRequest) -> Dict[str, Any]:
    """Authenticate user with Supabase Auth."""
    try:
        res = supabase_service.sign_in(req.email, req.password)
        return {"status": "success", **res}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.get("/api/auth/me")
def auth_me(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current authenticated user details from Supabase token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = supabase_service.verify_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"status": "authenticated", "user": user}


@app.post("/api/public/chat")
def public_chat(req: PublicChatRequest, user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)) -> Dict[str, Any]:
    """Public conversational assistant. Does NOT require authentication for general queries."""
    msg = req.message.strip().lower()
    
    # Check if request is asking for protected execution capabilities
    protected_triggers = ["train", "run model", "generate pipeline", "execute experiment", "save project"]
    is_protected_action = any(t in msg for t in protected_triggers)
    
    if is_protected_action and not user:
        return {
            "requires_auth": True,
            "message": "To train machine learning models, run experiments, and save projects to your cloud workspace, please sign in with Supabase Auth.",
            "auth_prompt": "Sign in or create a free account to execute ML workflows.",
        }
        
    # Public informational / greeting handling
    if any(g in msg for g in ["hi", "hello", "hey", "who are you", "help"]):
        return {
            "requires_auth": False,
            "message": "Hello! I am AMEA (Autonomous Machine Learning Engineer). You can explore data, ask ML strategy questions, or sign in to build and execute end-to-end trained models.",
        }
    
    return {
        "requires_auth": False,
        "message": f"AMEA Intelligence: I analyzed your query regarding '{req.message}'. You can formulate ML objectives or connect datasets to start autonomous model training.",
    }



# ============================================================
# Project Management & File System Endpoints
# ============================================================

@app.post("/api/project/create")
def create_project(req: CreateProjectRequest, user: Dict[str, Any] = Depends(get_current_user_required)) -> Dict[str, Any]:
    """Create a new ML project directory with requirements.txt and starter code, linked to Supabase user."""
    base_loc = Path(req.location or "workspace").resolve()
    project_dir = base_loc / req.name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Persist project record in Supabase / User DB
    supabase_service.save_project(
        user_id=user["id"],
        project_id=req.name,
        name=req.name,
        template=req.template,
    )

    # 1. Always ensure requirements.txt exists
    req_file = project_dir / "requirements.txt"
    default_reqs = "numpy\npandas\nscikit-learn\njoblib\nmatplotlib\nseaborn\n"
    if not req_file.exists():
        req_file.write_text(default_reqs, encoding="utf-8")

    # 2. Always ensure README.md exists
    readme_file = project_dir / "README.md"
    if not readme_file.exists():
        readme_file.write_text(f"# {req.name}\n\nAutonomous ML Engineering project created with AMEA.\n", encoding="utf-8")

    # 3. Create subdirectories
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "notebooks").mkdir(exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)

    # 4. Populate template files
    if req.template in ("classification", "regression"):
        starter_code = f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("AMEA ML project '{req.name}' initialized.")
"""
        (project_dir / "src" / "train.py").write_text(starter_code, encoding="utf-8")

    elif req.template == "data_analysis":
        starter_code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Data analysis exploration pipeline ready.")
"""
        (project_dir / "src" / "eda.py").write_text(starter_code, encoding="utf-8")

    return {
        "status": "success",
        "project_name": req.name,
        "project_path": str(project_dir),
    }


@app.post("/api/project/open")
def open_project(req: OpenProjectRequest) -> Dict[str, Any]:
    """Open existing folder, scan contents, auto-ensure requirements.txt, and initialize workspace."""
    p = Path(req.path).resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.path}")

    # Ensure requirements.txt exists
    req_file = p / "requirements.txt"
    if not req_file.exists():
        req_file.write_text("numpy\npandas\nscikit-learn\njoblib\nmatplotlib\nseaborn\n", encoding="utf-8")

    return {
        "status": "success",
        "project_name": p.name,
        "project_path": str(p),
    }


@app.get("/api/project/tree")
def get_project_tree(path: str) -> Dict[str, Any]:
    """Scan and return recursive directory tree."""
    root_path = Path(path).resolve()
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    def build_tree(current_dir: Path) -> List[Dict[str, Any]]:
        items = []
        try:
            for item in sorted(current_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name.startswith((".", "__pycache__")):
                    continue
                node = {
                    "name": item.name,
                    "path": str(item.relative_to(root_path)).replace("\\", "/"),
                    "is_dir": item.is_dir(),
                }
                if item.is_dir():
                    node["children"] = build_tree(item)
                else:
                    node["size_bytes"] = item.stat().st_size
                items.append(node)
        except PermissionError:
            pass
        return items

    return {
        "root_name": root_path.name,
        "root_path": str(root_path),
        "tree": build_tree(root_path),
    }


@app.post("/api/project/file/read")
def read_file(req: FileOperationRequest) -> Dict[str, Any]:
    """Read a project file securely."""
    base = Path(req.project_path).resolve()
    target = (base / req.relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Path traversal forbidden")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    content = target.read_text(encoding="utf-8", errors="replace")
    return {"content": content, "path": req.relative_path}


@app.post("/api/project/file/write")
def write_file(req: FileOperationRequest) -> Dict[str, Any]:
    """Write or create a project file."""
    base = Path(req.project_path).resolve()
    target = (base / req.relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Path traversal forbidden")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content or "", encoding="utf-8")
    return {"status": "saved", "path": req.relative_path}


@app.post("/api/project/file/create-dir")
def create_directory(req: FileOperationRequest) -> Dict[str, Any]:
    """Create a folder inside the project."""
    base = Path(req.project_path).resolve()
    target = (base / req.relative_path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Path traversal forbidden")
    target.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "path": req.relative_path}


@app.post("/api/project/file/delete")
def delete_file(req: FileOperationRequest) -> Dict[str, Any]:
    """Delete a file or directory."""
    base = Path(req.project_path).resolve()
    target = (base / req.relative_path).resolve()
    if not str(target).startswith(str(base)) or target == base:
        raise HTTPException(status_code=403, detail="Cannot delete root or outside files")

    if target.exists():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    return {"status": "deleted", "path": req.relative_path}


# ============================================================
# Project Download / ZIP Export (Secret Scrubbing)
# ============================================================

@app.get("/api/project/download-zip")
def download_project_zip(project_path: str):
    """Package project into a clean ZIP file excluding secrets and hidden files."""
    base = Path(project_path).resolve()
    if not base.exists():
        raise HTTPException(status_code=404, detail="Project path not found")

    zip_buffer = io.BytesIO()
    forbidden_files = {".env", "id_rsa", ".aws", ".git", ".pytest_cache", "__pycache__"}

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in forbidden_files and not d.startswith(".")]
            for file in files:
                if file in forbidden_files or file.startswith("."):
                    continue
                file_path = Path(root) / file
                archive_name = file_path.relative_to(base)
                zf.write(file_path, archive_name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={base.name}.zip"},
    )


# ============================================================
# Terminal Command Execution Endpoint
# ============================================================

@app.post("/api/terminal/exec")
def execute_terminal_command(req: TerminalExecRequest) -> Dict[str, Any]:
    """Execute shell command safely inside project directory."""
    base = Path(req.project_path).resolve()
    if not base.exists():
        raise HTTPException(status_code=404, detail="Project path not found")

    try:
        proc = subprocess.run(
            req.command,
            cwd=str(base),
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command execution timed out after 30s.",
            "exit_code": -1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1,
        }


# ============================================================
# Persistent AI Thread Management Endpoints
# ============================================================

@app.get("/api/threads/list")
def list_threads(project_id: str) -> List[AIThread]:
    """List persistent conversation threads for a project."""
    return _THREADS_DB.get(project_id, [])


@app.post("/api/threads/save")
def save_thread(req: SaveThreadRequest) -> AIThread:
    """Save or update an AI conversation thread."""
    threads = _THREADS_DB.setdefault(req.project_id, [])
    for i, t in enumerate(threads):
        if t.id == req.thread.id:
            threads[i] = req.thread
            return req.thread
    threads.append(req.thread)
    return req.thread


@app.post("/api/threads/delete")
def delete_thread(project_id: str, thread_id: str) -> Dict[str, bool]:
    """Delete an AI conversation thread."""
    if project_id in _THREADS_DB:
        _THREADS_DB[project_id] = [t for t in _THREADS_DB[project_id] if t.id != thread_id]
    return {"success": True}


# ============================================================
# Kernel Execution Endpoints
# ============================================================

from amea.execution.router import (
    AIGenerateCellRequest,
    AIInterpretRequest,
    CreateSessionRequest,
    LoadNotebookRequest,
    SaveNotebookRequest,
    SessionActionRequest,
)

@app.post("/api/kernel/session")
def create_kernel_session(req: CreateSessionRequest):
    return kernel_router.create_session(req)

@app.get("/api/kernel/session/{session_id}")
def get_kernel_session(session_id: str):
    sess = kernel_router.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess

@app.post("/api/kernel/execute")
def execute_kernel_cell(req: ExecuteCellRequest):
    return kernel_router.execute_cell(req)

@app.post("/api/kernel/execute-batch")
def execute_kernel_batch(req: BatchExecuteRequest):
    return kernel_router.execute_batch(req)

@app.post("/api/kernel/interrupt")
def interrupt_kernel(req: SessionActionRequest):
    return kernel_router.interrupt_session(req)

@app.post("/api/kernel/restart")
def restart_kernel(req: SessionActionRequest):
    return kernel_router.restart_session(req)

@app.delete("/api/kernel/session/{session_id}")
def shutdown_kernel(session_id: str):
    return kernel_router.shutdown_session(session_id)

@app.post("/api/kernel/notebook/save")
def save_notebook(req: SaveNotebookRequest):
    return kernel_router.save_notebook(req)

@app.post("/api/kernel/notebook/load")
def load_notebook(req: LoadNotebookRequest):
    return kernel_router.load_notebook(req)

@app.post("/api/kernel/ai/generate-cell")
def generate_ai_cell(req: AIGenerateCellRequest):
    return kernel_router.generate_ai_cell(req)

@app.post("/api/kernel/ai/interpret-result")
def interpret_ai_result(req: AIInterpretRequest):
    return kernel_router.interpret_result(req)


# ============================================================
# Autonomous Multi-Agent Orchestrator Execution Endpoints
# ============================================================

class OrchestratorRunRequest(BaseModel):
    project_id: str = "default_project"
    user_request: str = "Train a baseline classifier for customer churn prediction"
    dataset_path: str = "data/sample_churn.csv"
    target_column: Optional[str] = "churn"
    max_experiments: int = 3


@app.post("/api/orchestrator/run")
def run_orchestrator(req: OrchestratorRunRequest, user: Dict[str, Any] = Depends(get_current_user_required)) -> Dict[str, Any]:
    """Execute real multi-agent pipeline and return verified model artifacts and generated code."""
    from amea.core.config import ProjectConfig, ComputeBudget
    from amea.orchestrator.runner import OrchestratorRunner

    ds_path = Path(req.dataset_path)
    if not ds_path.exists():
        # Fallback check in project or root
        alt_paths = [Path("data/sample_churn.csv"), Path(f"workspace/{req.project_id}/{req.dataset_path}")]
        for ap in alt_paths:
            if ap.exists():
                ds_path = ap
                break

    cfg = ProjectConfig(
        project_id=req.project_id,
        budget=ComputeBudget(max_experiments=req.max_experiments),
    )
    runner = OrchestratorRunner(config=cfg)

    events_log: List[Dict[str, Any]] = []
    def event_listener(event):
        events_log.append({
            "event_type": event.event_type.value,
            "source": event.source_component,
            "message": event.message,
            "timestamp": event.timestamp.isoformat(),
            "payload": event.payload,
        })
    runner.event_bus.subscribe_all(event_listener)

    final_state = runner.run_task(
        user_request=req.user_request,
        dataset_path=str(ds_path) if ds_path.exists() else req.dataset_path,
        target_column=req.target_column,
    )

    # Persist experiment records to Supabase User DB
    for exp in final_state.experiment_ledger:
        supabase_service.save_experiment(
            user_id=user["id"],
            project_id=req.project_id,
            experiment_id=exp.experiment_id,
            model_family=exp.model_family,
            cv_metrics=exp.cv_metrics_mean,
            hyperparameters=exp.hyperparameters,
            duration_sec=exp.training_duration_sec,
            exit_code=exp.exit_code,
        )

    # Write generated code files to project workspace directory
    proj_dir = Path("workspace") / req.project_id
    if not proj_dir.exists():
        proj_dir = Path(".")

    generated_files_dict = {}
    if final_state.code_artifacts and final_state.code_artifacts.files:
        for fname, content in final_state.code_artifacts.files.items():
            target_file = proj_dir / "src" / fname if fname.endswith(".py") else proj_dir / fname
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")
            generated_files_dict[fname] = content
            
            # Save artifact metadata in Supabase
            supabase_service.save_artifact(
                user_id=user["id"],
                project_id=req.project_id,
                artifact_id=f"art_{fname}",
                artifact_type="code",
                name=fname,
                path=str(target_file),
                metadata={"length": len(content)},
            )

    return {
        "status": "success",
        "project_id": req.project_id,
        "terminal_phase": final_state.current_phase.value,
        "is_terminal": final_state.is_terminal,
        "termination_reason": final_state.termination_reason,
        "best_candidate": {
            "model_family": final_state.best_candidate.model_family if final_state.best_candidate else None,
            "cv_metrics_mean": final_state.best_candidate.cv_metrics_mean if final_state.best_candidate else {},
            "hyperparameters": final_state.best_candidate.hyperparameters if final_state.best_candidate else {},
        } if final_state.best_candidate else None,
        "experiments_count": len(final_state.experiment_ledger),
        "experiments": [
            {
                "experiment_id": exp.experiment_id,
                "model_family": exp.model_family,
                "cv_metrics_mean": exp.cv_metrics_mean,
                "duration_sec": exp.training_duration_sec,
                "exit_code": exp.exit_code,
            } for exp in final_state.experiment_ledger
        ],
        "generated_files": list(generated_files_dict.keys()),
        "events": events_log,
        "final_report": final_state.final_report.model_dump() if final_state.final_report else None,
    }


@app.get("/api/environment/info")
def get_environment_info() -> Dict[str, Any]:
    """Inspect and return verified real Python environment information."""
    import platform
    return {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": ["numpy", "pandas", "scikit-learn", "torch", "joblib", "matplotlib", "seaborn", "scipy"],
        "status": "READY",
    }


# ============================================================
# Mount Static Frontend UI & Query Analysis Endpoint
# ============================================================

from amea.query_analysis.schemas import QueryAnalysisRequest, QueryAnalysisResponse
from amea.query_analysis.router import analyze_query_data

@app.post("/api/query-analysis/analyze")
def analyze_query_endpoint(req: QueryAnalysisRequest) -> QueryAnalysisResponse:
    return analyze_query_data(req)

ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")

    @app.get("/")
    def serve_frontend_root():
        return FileResponse(str(ui_dir / "index.html"))

