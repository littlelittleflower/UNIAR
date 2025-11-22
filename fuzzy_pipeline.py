
# fuzzy_pipeline.py

import torch
import dgl
from typing import Dict, Optional, Tuple

from fuzzy_membership import (
    TrapezoidalFuzzy,
    TriangularFuzzy,
    FuzzyRelationMatrix,
    GaussianFuzzy,
    tnorm_mix,
)

def compute_delta_t(g: dgl.DGLGraph, node_time_key: str = "time", out_key: str = "delta_t") -> None:
    device = g.device
    if node_time_key not in g.ndata:
        g.edata[out_key] = torch.zeros(g.num_edges(), device=device)
        return
    src, dst = g.edges()
    ntime = g.ndata[node_time_key].float()
    dt = ntime[dst] - ntime[src]
    g.edata[out_key] = dt.to(device)

def compute_edge_attribute_scalar(
    g: dgl.DGLGraph,
    prefer_edge_keys = ("impact_km", "severity"),
    prefer_node_keys = ("impact_km", "severity"),
    reducer: str = "max",
    out_key: str = "attr_scalar",
) -> None:
    device = g.device
    for k in prefer_edge_keys:
        if k in g.edata:
            g.edata[out_key] = g.edata[k].float()
            return
    for k in prefer_node_keys:
        if k in g.ndata:
            src, dst = g.edges()
            vs = g.ndata[k].float()
            if reducer == "mean":
                es = 0.5 * (vs[src] + vs[dst])
            elif reducer == "max":
                es = torch.maximum(vs[src], vs[dst])
            else:
                es = torch.minimum(vs[src], vs[dst])
            g.edata[out_key] = es.to(device)
            return
    if "weight" in g.edata:
        g.edata[out_key] = g.edata["weight"].float()
    else:
        g.edata[out_key] = torch.zeros(g.num_edges(), device=device)

def compute_sr_stat(
    g: dgl.DGLGraph,
    rel_type_key: str = "rel_type",
    out_key: str = "sr_stat",
    global_num_rel: int = None,   # ★ 新增
) -> None:
    device = g.device
    # 推断 rel_type：若无 rel_type 而有双向 'type'，用全局关系数取模，避免 batch 稀疏导致 R'=1
    if rel_type_key not in g.edata and "type" in g.edata:
        if global_num_rel is None:
            R_guess = int(g.edata["type"].max().item()) + 1
        else:
            R_guess = int(global_num_rel)
        g.edata[rel_type_key] = (g.edata["type"] % R_guess).long()

    if rel_type_key not in g.edata:
        g.edata[out_key] = torch.full((g.num_edges(),), 1.0, device=device)
        return

    src, _ = g.edges()
    r = g.edata[rel_type_key].long()
    num_nodes = g.num_nodes()
    R = int(r.max().item()) + 1

    idx_sr = r + src.to(r.dtype) * R
    ones = torch.ones_like(idx_sr, dtype=torch.float32, device=device)

    sr_counts = torch.zeros(num_nodes * R, dtype=torch.float32, device=device)
    sr_counts.scatter_add_(0, idx_sr, ones)

    src_counts = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    src_counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32, device=device))

    denom = src_counts[src].clamp_min(1.0)
    stat = (sr_counts[idx_sr] / denom).clamp(0.0, 1.0)
    g.edata[out_key] = stat

def build_relation_coocurrence(
    g: dgl.DGLGraph,
    rel_type_key: str = "rel_type",
    global_num_rel: int = None,   # ★ 新增
) -> torch.Tensor:
    # 同上：优先使用全局关系数来从 'type' 推断 rel_type
    if rel_type_key not in g.edata and "type" in g.edata:
        if global_num_rel is None:
            R_guess = int(g.edata["type"].max().item()) + 1
        else:
            R_guess = int(global_num_rel)
        g.edata[rel_type_key] = (g.edata["type"] % R_guess).long()

    if rel_type_key not in g.edata:
        return torch.ones(1, 1, dtype=torch.float32, device=g.device)

    src, _ = g.edges()
    r = g.edata[rel_type_key].long()
    R = int(r.max().item()) + 1

    out_eids = [[] for _ in range(g.num_nodes())]
    for eid, u in enumerate(src.tolist()):
        out_eids[u].append(eid)

    Cooc = torch.zeros(R, R, dtype=torch.float32, device=g.device)
    for eids in out_eids:
        if not eids:
            continue
        rs = r[eids]
        for i in range(len(rs)):
            for j in range(len(rs)):
                Cooc[rs[i], rs[j]] += 1.0
    return Cooc


def attach_and_compute_fuzziness(
    g: dgl.DGLGraph,
    modules: Dict[str, torch.nn.Module],
    *,
    node_time_key: str = "time",
    attr_edge_keys = ("impact_km", "severity"),
    attr_node_keys = ("impact_km", "severity"),
    rel_type_key: str = "rel_type",
    global_num_rel: int = None,   # ★ 新增：从外部传入（比如 self.args.num_rel）
    use_mix: bool = True,
    tnorm: str = "product",
):
    device = g.device

    # 1) 先派生基础特征
    compute_delta_t(g, node_time_key=node_time_key, out_key="delta_t")
    compute_edge_attribute_scalar(g, prefer_edge_keys=attr_edge_keys, prefer_node_keys=attr_node_keys, out_key="attr_scalar")
    compute_sr_stat(g, rel_type_key=rel_type_key, out_key="sr_stat", global_num_rel=global_num_rel)
    Cooc = build_relation_coocurrence(g, rel_type_key=rel_type_key, global_num_rel=global_num_rel)

    # 2) 取可学习模块
    trap: TrapezoidalFuzzy   = modules.get("trap", None)
    tri:  TriangularFuzzy    = modules.get("tri", None)
    frm:  FuzzyRelationMatrix= modules.get("frm", None)
    gau:  GaussianFuzzy      = modules.get("gau", None)

    # 3) 计算隶属度（有则用，无则回退 weight）
    if trap is not None:
        mu_attr = trap(g.edata["attr_scalar"].to(next(trap.parameters()).device)).to(device)
    else:
        mu_attr = g.edata.get("weight", torch.ones(g.num_edges(), device=device))

    if tri is not None:
        mu_time = tri(g.edata["delta_t"].to(next(tri.parameters()).device)).to(device)
    else:
        mu_time = g.edata.get("weight", torch.ones(g.num_edges(), device=device))

    if frm is not None:
        Rf = frm.num_rel
        Rg = int(g.edata.get(rel_type_key, torch.zeros(g.num_edges(), device=device)).max().item()) + 1 if g.num_edges() > 0 else 1
        if Rf != Rg:
            # ✅ 不再 load_state_dict，直接临时新建一个尺寸匹配的 frm 做这次前向
            use_frm = FuzzyRelationMatrix(num_rel=Rg).to(next(frm.parameters()).device)
        else:
            use_frm = frm
        M = use_frm(Cooc.to(next(use_frm.parameters()).device))  # [Rg, Rg]
        r = g.edata.get(rel_type_key, torch.zeros(g.num_edges(), dtype=torch.long, device=device)).long().to(M.device)
        mu_rel = M[r, r].to(device)
    else:
        mu_rel = g.edata.get("weight", torch.ones(g.num_edges(), device=device))

    if gau is not None:
        mu_other = gau(g.edata["sr_stat"].to(next(gau.parameters()).device)).to(device)
    else:
        mu_other = g.edata.get("weight", torch.ones(g.num_edges(), device=device))

    # 4) 写回图
    g.edata["mu_attr"]  = mu_attr
    g.edata["mu_time"]  = mu_time
    g.edata["mu_rel"]   = mu_rel
    g.edata["mu_other"] = mu_other

    mu_mix = None
    if use_mix:
        mu_mix = tnorm_mix(mu_rel, mu_time, mu_attr, kind=tnorm).to(device)
        g.edata["mu_mix"] = mu_mix

    return mu_attr, mu_time, mu_rel, mu_other, mu_mix, Cooc
