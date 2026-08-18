"""Resource Manager allocating CPU and memory to concurrent workers."""

import threading
from typing import Dict
from amea.core.config import ComputeBudget
from amea.core.exceptions import ResourceConstraintError
from amea.task.model import ResourceRequirement


class ResourceManager:
    """Thread-safe resource allocator enforcing authorized compute budget."""

    def __init__(self, budget: ComputeBudget, detected_cpu_cores: int, detected_ram_gb: float):
        self.budget = budget
        self.detected_cpu_cores = detected_cpu_cores
        self.detected_ram_gb = detected_ram_gb

        # User-authorized caps (never exceeding detected physical capabilities)
        self.authorized_cpu = min(budget.max_cpu_cores, detected_cpu_cores)
        self.authorized_ram_mb = int(min(detected_ram_gb * 1024, 64 * 1024))

        self._allocated_cpu = 0
        self._allocated_ram_mb = 0
        self._lock = threading.Lock()

    def acquire(self, req: ResourceRequirement) -> bool:
        """Attempt to reserve resources for a worker run."""
        with self._lock:
            if (self._allocated_cpu + req.cpu_cores <= self.authorized_cpu and
                    self._allocated_ram_mb + req.ram_mb <= self.authorized_ram_mb):
                self._allocated_cpu += req.cpu_cores
                self._allocated_ram_mb += req.ram_mb
                return True
            return False

    def release(self, req: ResourceRequirement) -> None:
        """Release allocated resources when worker terminates."""
        with self._lock:
            self._allocated_cpu = max(0, self._allocated_cpu - req.cpu_cores)
            self._allocated_ram_mb = max(0, self._allocated_ram_mb - req.ram_mb)

    def get_utilization(self) -> Dict[str, float]:
        """Return current resource allocation stats."""
        with self._lock:
            return {
                "allocated_cpu": self._allocated_cpu,
                "authorized_cpu": self.authorized_cpu,
                "allocated_ram_mb": self._allocated_ram_mb,
                "authorized_ram_mb": self.authorized_ram_mb,
            }
