from __future__ import annotations

from server.cluster.base import ClusterBackend
from server.cluster.slurm import SlurmBackend
from server.cluster.local import LocalBackend


def get_backend(name: str) -> ClusterBackend:
    if name == "slurm":
        return SlurmBackend()
    return LocalBackend()
