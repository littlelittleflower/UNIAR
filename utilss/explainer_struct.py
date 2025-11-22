# utils/explainer_struct.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EdgeInfo:
    eid: int                     # DGL 边 id（hashable）
    mu_rel: float
    mu_time: float
    mu_attr: float
    time_ok: Optional[bool] = None

@dataclass
class PathInfo:
    edges: List[EdgeInfo]
    def length(self) -> int:
        return len(self.edges)
