# utils/abduction_explainer.py
import torch
import dgl
from typing import List, Sequence
from .explainer_struct import EdgeInfo, PathInfo

class AbductionExplainer:

    def __init__(self, max_hops:int=3, per_node_beam:int=8, tnorm:str="product", device="cuda"):
        self.g = None
        self.max_hops = max_hops         # 推荐 2 或 3
        self.per_node_beam = per_node_beam
        self.tnorm = tnorm
        self.device = device

    def set_context(self, dgl_graph: dgl.DGLHeteroGraph):
        self.g = dgl_graph
        for name in ["mu_rel","mu_time","mu_attr"]:
            if name not in self.g.edata:
                if "weight" in self.g.edata:
                    self.g.edata[name] = self.g.edata["weight"].clone()
                else:
                    self.g.edata[name] = torch.ones(self.g.num_edges(), device=self.g.device)

        if "eid" not in self.g.edata:

            self.g.edata["eid"] = torch.arange(self.g.num_edges(), device=self.g.device)

    # ---------- public APIs ----------
    def generate(self, tri: torch.Tensor, K:int=20) -> List[List[PathInfo]]:

        assert self.g is not None, "Call set_context(sup_graph) before generate(...)"
        h_list = tri[:,0].tolist()
        t_list = tri[:,2].tolist()

        paths_batch: List[List[PathInfo]] = []
        for h, t in zip(h_list, t_list):
            cand = self._k_hop_paths(h, t, max_hops=self.max_hops, topK=K)
            paths_batch.append(cand)
        return paths_batch

    def corrupt(self, paths_batch: List[List[PathInfo]]) -> List[List[PathInfo]]:

        neg_batch = []
        for paths in paths_batch:
            neg_paths = []
            for p in paths:
                if len(p.edges) == 0:
                    continue
                # 复制一条
                e_list = []
                for e in p.edges:
                    # 降低时间一致性，模拟违反时序
                    e_list.append(EdgeInfo(
                        eid=e.eid,
                        mu_rel=e.mu_rel,
                        mu_time=max(0.0, float(e.mu_time) * 0.3),   # time 衰减
                        mu_attr=e.mu_attr,
                        time_ok=False
                    ))
                neg_paths.append(PathInfo(edges=e_list))
            neg_batch.append(neg_paths if neg_paths else [])
        return neg_batch

    # ---------- internal search ----------

    def _build_path(self, eids:Sequence[int], mu_rel, mu_time, mu_attr, eid_all) -> PathInfo:
        edges = []
        for eid in eids:
            edges.append(EdgeInfo(
                eid=int(eid_all[eid].item()),
                mu_rel=float(mu_rel[eid]),
                mu_time=float(mu_time[eid]),
                mu_attr=float(mu_attr[eid]),
                time_ok=None
            ))
        return PathInfo(edges=edges)
