from utils import get_g_bidir,get_g_bidir_num
from datasets import TrainSubgraphDataset, ValidSubgraphDataset
from torch.utils.data import DataLoader
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch_geometric.data import Data
import os
from multi_view_rgcn import MultiViewRGCN
from fuzzy_membership import TrapezoidalFuzzy, TriangularFuzzy, FuzzyRelationMatrix, GaussianFuzzy
from fuzzy_pipeline import attach_and_compute_fuzziness
import torch
import json
import re, json, math, numpy as np, igraph
from typing import Dict, Any, List, Sequence, Optional
import pickle
from torch import optim
from trainer import Trainer
import dgl
import igraph
import numpy as np
from utilss.abduction_explainer import AbductionExplainer
import math
from collections import defaultdict as ddict

class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        self.gloal_embeddings = None
        self.test_model = None
        self.explainer = AbductionExplainer(max_hops=3, per_node_beam=8, tnorm="product", device=self.args.gpu)
        train_subgraph_dataset = TrainSubgraphDataset(args)
        valid_subgraph_dataset = ValidSubgraphDataset(args)
        self.train_subgraph_dataloader = DataLoader(train_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=True, collate_fn=TrainSubgraphDataset.collate_fn)
        self.valid_subgraph_dataloader = DataLoader(valid_subgraph_dataset, batch_size=args.metatrain_bs,
                                                    shuffle=False, collate_fn=ValidSubgraphDataset.collate_fn)

        self.global_R = len(self.args.relation2id)

        # 四类可学习隶属函数
        self.fuzzy_modules = {
            "trap": TrapezoidalFuzzy(init=(20., 50., 80., 120.)).to(self.args.gpu),  # 属性（梯形）
            "tri": TriangularFuzzy(init=(1., 3., 6.)).to(self.args.gpu),  # 时间（三角）
            "frm": FuzzyRelationMatrix(num_rel=self.global_R).to(self.args.gpu),  # 关系（共现+高斯）
            "gau": GaussianFuzzy(init_mean=0.5, init_sigma=0.25).to(self.args.gpu),  # 其他（高斯）
        }

        # optim
        self.optimizer = optim.Adam(list(self.ent_init.parameters()) + list(self.rgcn.parameters())
                                    + list(self.kge_model.parameters())+list(self.multiviewRGCN.parameters())+list(self.GatedFusion.parameters()), lr=self.args.metatrain_lr)

    def sample_hard_negatives(self, que_tri, hr2t, rt2h, ent_emb, num_neg=10):
        """
        生成 Hard Negatives：
        - 使用实体嵌入找到最接近正样本的负样本
        """
        que_neg_tail_ent = []
        que_neg_head_ent = []

        entity_ids = np.arange(ent_emb.shape[0])  # 所有实体索引
        ent_emb_np = ent_emb.cpu().detach().numpy()  # 转换为 NumPy

        for h, r, t in que_tri:
            # 候选负样本（排除正样本）
            h, r, t = int(h.item()), int(r.item()), int(t.item())  # 先转换为 int
            tail_candidates = np.delete(entity_ids, hr2t[(h, r)])
            head_candidates = np.delete(entity_ids, rt2h[(r, t)])

            # 计算负样本与正样本的相似性（欧几里得距离）
            tail_dists = np.linalg.norm(ent_emb_np[tail_candidates] - ent_emb_np[t], axis=1)
            head_dists = np.linalg.norm(ent_emb_np[head_candidates] - ent_emb_np[h], axis=1)

            # 选择 Hard Negatives
            tail_sorted = tail_candidates[np.argsort(tail_dists)[:num_neg]]
            head_sorted = head_candidates[np.argsort(head_dists)[:num_neg]]

            que_neg_tail_ent.append(tail_sorted)
            que_neg_head_ent.append(head_sorted)

        return torch.tensor(que_neg_tail_ent), torch.tensor(que_neg_head_ent)


    def load_pretrain(self):
        state = torch.load(self.args.pretrain_state, map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])


    def merge_prompt_and_structure_numpy(self,prompt_graph, A_struct, type_for_structure=4):
        """
        将 prompt PyG 图和结构邻接矩阵 A_struct 合并为一个 NumPy 邻接矩阵。

        Args:
            prompt_graph: PyG Data 对象（包含 edge_index 和 edge_type）
            A_struct: numpy.ndarray，形状为 [num_rel, num_rel]，结构连接图
            type_for_structure: int，结构图边赋予的类型编号（如 4）

        Returns:
            A_merged: numpy.ndarray，形状为 [num_rel, num_rel]，合并后的邻接矩阵
        """
        num_rel = A_struct.shape[0]
        A_merged = np.zeros((num_rel, num_rel), dtype=int)

        # 1️⃣ 保留原结构图边（标为 4）
        A_merged[A_struct > 0] = type_for_structure

        # 2️⃣ 添加 prompt 图边（用原来的 edge_type）
        edge_index = prompt_graph.edge_index.cpu().numpy()
        edge_type = prompt_graph.edge_type.cpu().numpy()

        for idx in range(edge_index.shape[1]):
            src = edge_index[0, idx]
            dst = edge_index[1, idx]
            etype = edge_type[idx]
            A_merged[src, dst] = etype  # 提示图优先覆盖结构图（可调）

        return A_merged

    def build_prompt_graph_from_llm(self,llm_dict):
        """
        从 LLM JSON 结构构建 Gp 提示图三元组集合
        """
        from collections import defaultdict
        type2rel_head = defaultdict(set)
        type2rel_tail = defaultdict(set)

        for rel_name, info in llm_dict.items():
            for t in info.get("head", []):
                type2rel_head[t].add(rel_name)
            for t in info.get("tail", []):
                type2rel_tail[t].add(rel_name)

        edges = set()

        for type_, rels in type2rel_head.items():
            rels = list(rels)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    edges.add((rels[i], "h2h", rels[j]))

        for type_, rels in type2rel_tail.items():
            rels = list(rels)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    edges.add((rels[i], "t2t", rels[j]))

        for type_ in set(type2rel_head.keys()).intersection(type2rel_tail.keys()):
            for h in type2rel_head[type_]:
                for t in type2rel_tail[type_]:
                    edges.add((h, "h2t", t))

        return {
            "T": edges,
            "R": set([h for h, _, _ in edges] + [t for _, _, t in edges]),
            "Rfund": set([h for h, _, _ in edges]),
            "relation_edges": edges
        }

    def generate_relation_triplets_llm(self,triplet, num_ent, num_rel, B,relation_graph2):
        """
        生成关系三元组 (head_relation, tail_relation, weight)
        """
        # 1️⃣ **构造关系-关系邻接矩阵 A**
        A = self.create_relation_graph(triplet, num_ent, num_rel)

        A = self.merge_prompt_and_structure_numpy(relation_graph2, A)

        # 2️⃣ **将邻接矩阵转换为加权图**
        G_rel = igraph.Graph.Weighted_Adjacency(A.tolist())

        # 3️⃣ **从 G_rel 提取关系三元组**
        relation_triplets = self.get_relation_triplets(G_rel, B)

        return relation_triplets

    def get_relation_triplets(self,G_rel, B):
        rel_triplets = []
        for tup in G_rel.get_edgelist():
            h, t = tup
            tupid = G_rel.get_eid(h, t)
            w = G_rel.es[tupid]["weight"]
            rel_triplets.append((int(h), int(t), float(w)))
        rel_triplets = np.array(rel_triplets)

        nnz = len(rel_triplets)  # 关系三元组的数量

        temp = (-rel_triplets[:, 2]).argsort()
        weight_ranks = np.empty_like(temp)
        weight_ranks[temp] = np.arange(nnz) + 1

        relation_triplets = []
        for idx, triplet in enumerate(rel_triplets):
            h, t, w = triplet
            rk = int(math.ceil(weight_ranks[idx] / nnz * B)) - 1
            relation_triplets.append([int(h), int(t), rk])
            assert rk >= 0
            assert rk < B

        return np.array(relation_triplets)

    def _ekg_to_text_multiview(self,
            g_list: List[dgl.DGLGraph],
            view_names: List[str],
            max_nodes_per_view: int = 200,
            max_edges_per_view: int = 500,
    ) -> str:
        """
        将 EE/EV/VV 三个图拼接成带 VIEW 标记的文本：
        - NODE 行:  VIEW  node_id  Type=Event|Entity|Unknown
        - EDGE 行:  VIEW  src  rel  tgt  mu=...
        需要 g.ndata['node_type'] (事件=1/实体=0)；若没有，就标 Unknown。
        """
        out_lines = []
        for g, vname in zip(g_list, view_names):
            N = g.num_nodes();
            E = g.num_edges()
            take_nodes = min(N, max_nodes_per_view)
            take_edges = min(E, max_edges_per_view)

            node_type = g.ndata.get("node_type", None)
            for n in range(take_nodes):
                if node_type is None:
                    out_lines.append(f"NODE\t{vname}\t{n}\tType=Unknown")
                else:
                    typ = int(node_type[n].item())
                    out_lines.append(f"NODE\t{vname}\t{n}\tType={'Event' if typ == 1 else 'Entity'}")

            src, dst = g.edges()
            w = g.edata.get("weight", torch.ones(E, device=g.device))
            # 优先用 rel_name（字符串），否则用 rel_type（id）
            rel_name = g.edata.get("rel_name", None)
            rel_type = g.edata.get("rel_type", None)

            for e in range(take_edges):
                u = int(src[e].item());
                v = int(dst[e].item());
                mu = float(w[e].item())
                if rel_name is not None:
                    r = rel_name[e]
                    r = int(r.item()) if torch.is_tensor(r) else r
                elif rel_type is not None:
                    r = int(rel_type[e].item())
                else:
                    r = "Rel"
                out_lines.append(f"EDGE\t{vname}\t{u}\t{r}\t{v}\tmu={mu:.3f}")

        return "\n".join(out_lines)

    # ====== 2) 构造 Prompt（强制只输出 JSON）======
    def _build_prompt_all_relations_multiview(self,snippet: str, allowed_labels: Sequence[str]) -> str:
        schema = {"type": "object", "properties": {"edges": {"type": "array"}}, "required": ["edges"]}
        labs = ", ".join(allowed_labels)
        prompt = f"""
    You are an expert annotator. Convert the following fuzzy EKG (3 views: EE, EV, VV) into a unified fuzzy edge list.

    STRICT OUTPUT RULES:
    - Return ONLY valid JSON (no markdown, no explanation).
    - The FIRST character must be '{{' and the LAST character must be '}}'.
    - JSON schema: {json.dumps(schema, ensure_ascii=False)}
    - Each item in "edges" MUST be:
      {{"src":int, "tgt":int, "label":str, "mu":float, "src_type":"Event|Entity|Unknown", "tgt_type":"Event|Entity|Unknown"}}
    - "mu" must be in [0,1].
    - "label" must be one of: [{labs}]
    - Prefer short/high-confidence edges and avoid duplicates.

    EKG (three views merged, lines start with NODE/EDGE and a VIEW tag):
    {snippet}
    """
        return prompt.strip()

    # ====== 3) 稳健解析 LLM 输出 ======
    def _extract_json_block(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"<json>\s*([\s\S]*?)\s*</json>", text, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        l = text.find("{");
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            cand = text[l:r + 1]
            try:
                return json.loads(cand)
            except Exception:
                pass
        raise ValueError("LLM did not return valid JSON. Got:\n" + text[:800])

    def _parse_llm_edges_any(self,text: str, allowed_labels: Sequence[str]) -> List[Dict[str, Any]]:
        js = self._extract_json_block(text)
        edges = []
        if "edges" in js and isinstance(js["edges"], list):
            for e in js["edges"]:
                try:
                    src = int(e["src"]);
                    tgt = int(e["tgt"])
                    lab = str(e["label"]);
                    mu = float(e.get("mu", 0.5))
                    st = str(e.get("src_type", "Unknown"))
                    tt = str(e.get("tgt_type", "Unknown"))
                    if lab not in allowed_labels:
                        continue
                    mu = float(max(0.0, min(1.0, mu)))
                    if st not in ("Event", "Entity", "Unknown") or tt not in ("Event", "Entity", "Unknown"):
                        continue
                    edges.append({"src": src, "tgt": tgt, "label": lab, "mu": mu, "src_type": st, "tgt_type": tt})
                except Exception:
                    continue
        return edges

    # ====== 4) 仅用 LLM 边累计成关系—关系邻接矩阵 A ======
    def _accumulate_llm_edges_to_A_any(self,
            edges: List[Dict[str, Any]],
            relation2id: Dict[str, int],
            num_rel: int,
    ) -> np.ndarray:
        """
        策略：对每个节点 u，收集它所有“出边”的关系类型（以 mu 为权），
        对任意两个出边关系 (ri, rj) 形成有序对累加权重（权重=两边 mu 的均值；可改为乘积/最小值）。
        """
        buckets = {}  # node -> List[(rel_id, mu)]
        for e in edges:
            lab = e["label"]
            if lab not in relation2id:
                continue
            rid = int(relation2id[lab])
            mu = float(e["mu"])
            u = int(e["src"])
            buckets.setdefault(u, []).append((rid, mu))

        A = np.zeros((num_rel, num_rel), dtype=np.float32)
        for _, lst in buckets.items():
            Rk = [rid for (rid, _) in lst]
            Wk = [mu for (_, mu) in lst]
            for i in range(len(Rk)):
                for j in range(len(Rk)):
                    ri, rj = Rk[i], Rk[j]
                    wij = 0.5 * (Wk[i] + Wk[j])  # 可换成 Wk[i]*Wk[j] 或 min(...)
                    A[ri, rj] += wij
        return A

    # ====== 5) 主函数：纯 LLM，多视图 ======
    def generate_relation_triplets_llm_only_multiview(
            self,
            num_ent: int,
            num_rel: int,
            B: int,
            *,
            g1: dgl.DGLGraph,  # EE
            g2: dgl.DGLGraph,  # EV
            g3: dgl.DGLGraph,  # VV
            llm,  # 你的 LLM 对象：HuggingFaceLLMGPTQ / HTTPChatLLM / 其他实现 .generate
            relation2id: Dict[str, int],  # 全局“关系名→id”（必须覆盖 LLM 可能输出的所有标签）
            allowed_labels: Optional[Sequence[str]] = None,  # 如果不传，默认用 relation2id.keys()
            max_nodes_per_view: int = 200,
            max_edges_per_view: int = 500,
            temperature: float = 0.3,
            max_new_tokens: int = 512,
    ) -> np.ndarray:
        """
        仅使用大模型：将 EE/EV/VV 三个视图一起作为上下文，生成覆盖所有类型边的列表，
        再汇总成关系—关系邻接矩阵 A，最后输出 [h_rel, t_rel, rk]（与原函数一致）。
        """
        if allowed_labels is None:
            allowed_labels = list(relation2id.keys())

        # A) 构造成熟的多视图文本
        snippet = self._ekg_to_text_multiview(
            [g1, g2, g3],
            view_names=["EE", "EV", "VV"],
            max_nodes_per_view=max_nodes_per_view,
            max_edges_per_view=max_edges_per_view,
        )
        prompt = self._build_prompt_all_relations_multiview(snippet, allowed_labels)

        # B) 调用 LLM（尽量只返回补全；若回显了 prompt，裁掉）
        text = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # C) 解析 LLM 边（覆盖 EE/EV/VV）
        llm_edges = self._parse_llm_edges_any(text, allowed_labels=allowed_labels)

        # D) 完全不做统计，仅用 LLM 边累计成 A
        A = self._accumulate_llm_edges_to_A_any(llm_edges, relation2id, num_rel)

        # E) 产出与你原版一致的 relation_triplets: [h_rel, t_rel, rk]
        if A.sum() == 0:
            return np.array([[0, 0, 0]], dtype=np.int64)

        G_rel = igraph.Graph.Weighted_Adjacency(A.tolist(), mode=igraph.ADJ_DIRECTED, attr="weight", loops=True)

        rel_triplets = []
        for (h, t) in G_rel.get_edgelist():
            eid = G_rel.get_eid(h, t)
            w = G_rel.es[eid]["weight"]
            rel_triplets.append((int(h), int(t), float(w)))
        rel_triplets = np.array(rel_triplets, dtype=np.float32)

        nnz = len(rel_triplets)
        order = (-rel_triplets[:, 2]).argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(nnz) + 1  # 1..nnz

        out = []
        for idx, (h, t, w) in enumerate(rel_triplets):
            rk = int(math.ceil(ranks[idx] / nnz * B)) - 1
            rk = max(0, min(B - 1, rk))
            out.append([int(h), int(t), rk])
        return np.array(out, dtype=np.int64)

    # -------- 1) Prompt：要求 LLM 直接输出包含 EE/EV/VV 的边 --------
    def _build_prompt_all_relations(self,ekg_snippet,allowed_labels):
        """
        让 LLM 在一个 JSON 里输出覆盖 EE/EV/VV 的边列表：
        edges: [{"src":int,"tgt":int,"label":str,"mu":float,"src_type":"Event|Entity","tgt_type":"Event|Entity"}]
        """
        schema = {
            "type": "object",
            "properties": {
                "edges": {"type": "array"}
            },
            "required": ["edges"]
        }
        labs = ", ".join(allowed_labels)
        prompt = f"""
    You are an expert annotator. Convert the following fuzzy EKG into a unified fuzzy edge list covering:
    - EE (event→event), EV (event→entity), VV (entity→entity).

    STRICT OUTPUT RULES:
    - Return ONLY valid JSON (no markdown, no text).
    - The FIRST character must be '{{' and the LAST character must be '}}'.
    - Use this schema: {json.dumps(schema, ensure_ascii=False)}
    - Each item in "edges" MUST be:
      {{"src":int, "tgt":int, "label":str, "mu":float, "src_type":"Event|Entity", "tgt_type":"Event|Entity"}}
    - "mu" must be in [0,1].
    - "label" must be one of: [{labs}]
    - Prefer short/high-confidence edges and avoid duplicates.

    EKG:
    {ekg_snippet}
    """
        return prompt.strip()

    # -------- 2) 解析：稳健提取 JSON（容忍模型偶尔夹带文字） --------
    def _extract_json_block(self,text: str) -> dict:
        # 直接 JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # ```json ... ```
        m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # <json> ... </json>
        m = re.search(r"<json>\s*([\s\S]*?)\s*</json>", text, re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 最大 {...}
        l = text.find("{");
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            cand = text[l:r + 1]
            try:
                return json.loads(cand)
            except Exception:
                pass
        raise ValueError("LLM did not return valid JSON. Got:\n" + text[:800])

    def _parse_llm_edges_any(self,text: str, allowed_labels):
        js = self._extract_json_block(text)
        edges = []
        if "edges" in js and isinstance(js["edges"], list):
            for e in js["edges"]:
                try:
                    src = int(e["src"]);
                    tgt = int(e["tgt"])
                    lab = str(e["label"])
                    mu = float(e.get("mu", 0.5))
                    st = str(e.get("src_type", "Event"))
                    tt = str(e.get("tgt_type", "Event"))
                    if lab not in allowed_labels:
                        continue
                    mu = float(max(0.0, min(1.0, mu)))
                    # 可选校验类型
                    if st not in ("Event", "Entity") or tt not in ("Event", "Entity"):
                        continue
                    edges.append({"src": src, "tgt": tgt, "label": lab, "mu": mu, "src_type": st, "tgt_type": tt})
                except Exception:
                    continue
        return edges

    # -------- 3) 将 LLM 边（任何类型）累计为 关系—关系邻接 A（纯 LLM） --------
    def _accumulate_llm_edges_to_A_any(self,edges,relation2id,num_rel):
        """
        策略：对每个节点 u，把其所有“出边”的关系类型汇总成一个桶，
        然后对 (ri,rj) 有序对累计权重（权重=两边 mu 的均值；可改乘法/最小值）。
        只用 LLM 的边，不做任何统计补充。
        """
        buckets = {}  # node -> List[(rel_id, mu)]
        for e in edges:
            lab = e["label"]
            if lab not in relation2id:
                continue
            rid = int(relation2id[lab])
            mu = float(e["mu"])
            u = int(e["src"])
            buckets.setdefault(u, []).append((rid, mu))

        A = np.zeros((num_rel, num_rel), dtype=np.float32)
        for _, lst in buckets.items():
            Rk = [rid for (rid, _) in lst]
            Wk = [mu for (_, mu) in lst]
            for i in range(len(Rk)):
                for j in range(len(Rk)):
                    ri, rj = Rk[i], Rk[j]
                    wij = 0.5 * (Wk[i] + Wk[j])
                    A[ri, rj] += wij
        return A

    # -------- 4) 主函数：只用 LLM 生成关系图，输出与你原版一致 --------
    def generate_relation_triplets_llm_only(self,
                                            num_ent: int,
                                            num_rel: int,
                                            B: int,
                                            *,
                                            g_ekg,  # DGL 大图（含节点类型/边权可选）
                                            llm,  # 任何实现了 .generate(prompt,...) 的对象（如 HuggingFaceLLMGPTQ/HTTPChatLLM）
                                            relation2id,  # 全局 关系名->id（需包含 EE/EV/VV 所有可能标签）
                                            allowed_labels,
                                            max_nodes: int = 200,
                                            max_edges: int = 500,
                                            temperature: float = 0.3,
                                            max_new_tokens: int = 512,
                                            ):
        """
        纯 LLM：从 EKG 文本片段生成覆盖 EE/EV/VV 的边，累计为关系—关系邻接 A，并输出 [h_rel, t_rel, rk]。
        - 不依赖任何人工统计/共现。
        - 返回格式与原 generate_relation_triplets 完全一致。
        """

        # A) 把 EKG 转成文本（可复用你已有的函数；此处给出一个最简兼容版）
        def ekg_to_text(g, max_nodes, max_edges):
            N = g.num_nodes();
            E = g.num_edges()
            take_nodes = min(N, max_nodes);
            take_edges = min(E, max_edges)
            src, dst = g.edges()
            lines = []
            node_type = g.ndata.get("node_type", None)  # 事件=1，实体=0（若无则 Unknown）
            for n in range(take_nodes):
                if node_type is None:
                    lines.append(f"NODE\\t{n}\\tType=Unknown")
                else:
                    typ = int(node_type[n].item())
                    lines.append(f"NODE\\t{n}\\tType={'Event' if typ == 1 else 'Entity'}")
            w = g.edata.get("weight", None)
            rel_name = g.edata.get("rel_type", None)
            for e in range(min(E, max_edges)):
                u = int(src[e].item());
                v = int(dst[e].item())
                mu = float(w[e].item()) if w is not None else 0.5
                if rel_name is not None:
                    r = int(rel_name[e].item())
                    lines.append(f"EDGE\\t{u}\\t{r}\\t{v}\\tmu={mu:.3f}")
                else:
                    lines.append(f"EDGE\\t{u}\\tREL\\t{v}\\tmu={mu:.3f}")
            return "\\n".join(lines)

        snippet = ekg_to_text(g_ekg, max_nodes=max_nodes, max_edges=max_edges)
        prompt = self._build_prompt_all_relations(snippet, allowed_labels)

        # B) 生成（只返回补全；若管道会回显输入，可手动裁剪）
        text = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        # C) 解析 LLM 返回的 edges
        llm_edges = self._parse_llm_edges_any(text, allowed_labels=allowed_labels)

        # D) 仅用 LLM 边累计成 A（覆盖 EE/EV/VV）
        A = self._accumulate_llm_edges_to_A_any(llm_edges, relation2id, num_rel)

        # E) 生成 igraph 加权图 → 输出 [h_rel, t_rel, rk]，与原版一致
        if A.sum() == 0:
            # 没产生任何边，兜底：返回一个最小条目
            return np.array([[0, 0, 0]], dtype=np.int64)

        G_rel = igraph.Graph.Weighted_Adjacency(A.tolist(), mode=igraph.ADJ_DIRECTED, attr="weight", loops=True)

        rel_triplets = []
        for (h, t) in G_rel.get_edgelist():
            eid = G_rel.get_eid(h, t)
            w = G_rel.es[eid]["weight"]
            rel_triplets.append((int(h), int(t), float(w)))
        rel_triplets = np.array(rel_triplets, dtype=np.float32)

        nnz = len(rel_triplets)
        order = (-rel_triplets[:, 2]).argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(nnz) + 1  # 1..nnz

        out = []
        for idx, (h, t, w) in enumerate(rel_triplets):
            rk = int(math.ceil(ranks[idx] / nnz * B)) - 1
            rk = max(0, min(B - 1, rk))
            out.append([int(h), int(t), rk])
        return np.array(out, dtype=np.int64)

    def create_relation_graph(self,triplet, num_ent, num_rel):
        """
        构造关系-关系的邻接矩阵 A，表示关系之间的连接强度
        """
        ind_h = triplet[:, :2]  # (h, r)
        ind_t = triplet[:, 1:]  # (r, t)

        # 1️⃣ **创建邻接矩阵 (num_rel × num_rel)，初始化为 0**
        A = np.zeros((num_rel, num_rel))

        # 2️⃣ **构造两个矩阵 E_h, E_t**
        E_h = np.zeros((num_ent, 2 * num_rel))
        E_t = np.zeros((num_ent, 2 * num_rel))

        for h, r in ind_h:
            E_h[h, r] = 1  # 头实体连接的关系

        for r, t in ind_t:
            E_t[t, r] = 1  # 尾实体连接的关系

        # 3️⃣ **计算邻接矩阵 A**
        D_h_inv = np.diag(1 / (E_h.sum(axis=1) + 1e-6))  # 归一化
        D_t_inv = np.diag(1 / (E_t.sum(axis=1) + 1e-6))

        A_h = E_h.T @ D_h_inv @ E_h  # 计算关系-关系连接
        A_t = E_t.T @ D_t_inv @ E_t

        A = A_h + A_t  # 关系-关系邻接矩阵
        return A

    def generate_relation_triplets(self,triplet, num_ent, num_rel, B):
        """
        生成关系三元组 (head_relation, tail_relation, weight)
        """
        # 1️⃣ **构造关系-关系邻接矩阵 A**
        A = self.create_relation_graph(triplet, num_ent, num_rel)

        # 2️⃣ **将邻接矩阵转换为加权图**
        G_rel = igraph.Graph.Weighted_Adjacency(A.tolist())

        # 3️⃣ **从 G_rel 提取关系三元组**
        relation_triplets = self.get_relation_triplets(G_rel, B)

        return relation_triplets

    def is_event(self,node_str,event2id):
        """判断是否为事件节点"""
        return node_str in event2id

    def get_node_id(self,node, entity2id, event2id, offset):
        if self.is_event(node,event2id):
            return event2id[node] + offset
        else:
            return entity2id[node]

    def get_relation_emb(self, relation_triplets):
        """
        使用生成的关系三元组构造关系图，并通过 RGCN 计算增强的关系嵌入。
        假设 relation_triplets 的形状为 (num_edges, 3)，
        其中第一列是 head_relation 索引，第二列是 tail_relation 索引，
        第三列存放边的类型信息（例如 rank）。
        """
        # 将关系三元组转换为 torch tensor（确保数据类型正确）
        relation_triplets = torch.tensor(relation_triplets, dtype=torch.long).to(self.args.gpu)
        src = relation_triplets[:, 0]  # head_relation
        dst = relation_triplets[:, 1]  # tail_relation

        # 构造 DGLGraph；节点代表所有关系，边由 src, dst 构成
        #就是单向的
        g_rel = dgl.graph((src, dst), num_nodes=self.args.num_rel*2).to(self.args.gpu)

        # 初始化节点特征，注意这里使用 self.args.rel_dim 作为特征维度（可根据你的设置调整）
        num_nodes = g_rel.num_nodes()
        # 这里初始化为全零，也可以使用随机初始化
        g_rel.ndata['feat'] = torch.zeros((num_nodes, self.args.rel_dim), device=self.args.gpu)

        # 设置边特征。这里将第三列作为边类型
        g_rel.edata['type'] = relation_triplets[:, 2]

        # 由于 RGCNLayer 内部会自动计算 g.in_degrees()，这里不必手动设置

        # 将构造好的关系图传入 RGCN 模型（你提供的 RGCN 模型代码已经支持 homogeneous graph 的输入）
        g_rel.ndata['h'] = torch.randn(num_nodes, self.args.ent_dim).to(self.args.gpu)
        # g_rel.ndata['h'] = self.ent_init(torch.arange(g_rel.num_nodes(), device=self.args.gpu))#??
        rel_emb = self.rgcn(g_rel)
        # print(f"rel_emb grad status: {rel_emb.requires_grad}")  # True 表示可训练
        return rel_emb

    def build_pyg_graph_from_triples(self,triples, rel_set, device='cuda'):
        type_map = {"h2h": 0, "t2t": 1, "h2t": 2, "t2h": 3}
        rel_id_map = {r: i for i, r in enumerate(sorted(rel_set))}
        edge_index = torch.tensor([[rel_id_map[h], rel_id_map[t]] for (h, _, t) in triples], dtype=torch.long).T
        edge_type = torch.tensor([type_map[r] for (_, r, _) in triples])
        return Data(edge_index=edge_index.to(device), edge_type=edge_type.to(device), num_nodes=len(rel_id_map),
                    num_relations=4)


    def generate_relation_triplets_llm(self,triplet, num_ent, num_rel, B,relation_graph2):
        """
        生成关系三元组 (head_relation, tail_relation, weight)
        """
        # 1️⃣ **构造关系-关系邻接矩阵 A**
        A = self.create_relation_graph(triplet, num_ent, num_rel)

        A = self.merge_prompt_and_structure_numpy(relation_graph2, A)

        # 2️⃣ **将邻接矩阵转换为加权图**
        G_rel = igraph.Graph.Weighted_Adjacency(A.tolist())

        # 3️⃣ **从 G_rel 提取关系三元组**
        relation_triplets = self.get_relation_triplets(G_rel, B)

        return relation_triplets

    def build_dgl_graph(self,edge_list,args,label='test'):
        # offset = len(entity2id)
        relation2id = args.relation2id
        if label=='train':
            offset = len(args.entity2id)
            entity2id = args.entity2id
            event2id = args.event2id
            fu_dict = args.fuzzy_dict
        else:
            offset = len(args.test_entity2id)
            entity2id = args.test_entity2id
            event2id = args.test_event2id
            fu_dict = args.test_fuzzy_dict
        src_ids, dst_ids, rel_ids,fuzziness_list = [], [], [],[]
        rel_map = {}#重新映射了？
        rel_id_counter = 0
        for h, t, r in edge_list:
            h_id = self.get_node_id(h, entity2id, event2id, offset)
            t_id = self.get_node_id(t, entity2id, event2id, offset)
            # t_id = int(t)
            src_ids.append(h_id)
            dst_ids.append(t_id)
            if r not in rel_map:
                rel_map[r] = rel_id_counter
                rel_id_counter += 1
            #模糊度的获取依旧是之前的映射
            fuzziness_list.append(fu_dict.get((h_id, relation2id[r], t_id)))
            #id为所谓，不用这个映射
            rel_ids.append(rel_map[r])

        g = dgl.graph((src_ids, dst_ids), num_nodes=len(entity2id) + len(event2id))
        g.edata['rel_type'] = torch.tensor(rel_ids)
        g.edata['weight'] = torch.tensor(fuzziness_list)
        return g

    def build_prompt_graph_from_llm_filtered(self,llm_dict, relation2id):
        """
        从 LLM JSON 中仅选出 relation2id 中的关系，构建 prompt graph。

        Args:
            llm_dict: LLM 生成的全量 JSON 提示词典
            relation2id: dict, 仅保留的关系名 → id

        Returns:
            Gp: dict with T, R, Rfund, relation_edges
        """
        from collections import defaultdict
        type2rel_head = defaultdict(set)
        type2rel_tail = defaultdict(set)

        # 筛选出 relation2id 中的关系
        for rel_name in relation2id:
            info = llm_dict.get(rel_name, {})
            for t in info.get("head", []):
                type2rel_head[t].add(rel_name)
            for t in info.get("tail", []):
                type2rel_tail[t].add(rel_name)

        # 构建三元组 edges
        edges = set()

        for type_, rels in type2rel_head.items():
            rels = list(rels)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    edges.add((rels[i], "h2h", rels[j]))

        for type_, rels in type2rel_tail.items():
            rels = list(rels)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    edges.add((rels[i], "t2t", rels[j]))

        for type_ in set(type2rel_head.keys()).intersection(type2rel_tail.keys()):
            for h in type2rel_head[type_]:
                for t in type2rel_tail[type_]:
                    edges.add((h, "h2t", t))

        return {
            "T": edges,
            "R": set([h for h, _, _ in edges] + [t for _, _, t in edges]),
            "Rfund": set([h for h, _, _ in edges]),
            "relation_edges": edges
        }


    def get_entity_emb(self, batch_sup_g, rel_emb):
        """
        使用 R-GCN 计算增强的实体嵌入。
        :param batch_sup_g: (DGLGraph) 支持集图
        :param rel_emb: (Tensor) 关系嵌入
        :return: (Tensor) 实体嵌入
        """
        # 1️⃣ **初始化节点特征**
        num_nodes = batch_sup_g.num_nodes()
        batch_sup_g.ndata['feat'] = torch.zeros((num_nodes, self.args.ent_dim), device=self.args.gpu)

        # 2️⃣ **设置边特征（关系嵌入）**
        batch_sup_g.edata['rel_emb'] = rel_emb[batch_sup_g.edata['type']]

        # 3️⃣ **调用 R-GCN 传播关系信息**
        ent_emb = self.rgcn(batch_sup_g)
        # print(f"ent_emb grad status: {ent_emb.requires_grad}")  # True 表示可训练

        return ent_emb

    def train(self):
        step = 0
        best_step = 0
        best_eval_rst = {
            'overall': {'mrr': 0.0, 'hits@1': 0.0, 'hits@5': 0.0, 'hits@10': 0.0},
            'event_query': {'mrr': 0.0, 'hits@1': 0.0, 'hits@5': 0.0, 'hits@10': 0.0},
            'entity_query': {'mrr': 0.0, 'hits@1': 0.0, 'hits@5': 0.0, 'hits@10': 0.0}
        }
        bad_count = 0
        self.logger.info('start meta-training')

        dataset_root = './data'
        json_path = os.path.join(dataset_root, "eventkg", 'eventkg-r2t-llama13b-des-fixed.json')
        with open(json_path, "r", encoding="utf-8") as f:
            llm_rel_dict = json.load(f)
        # Gp = self.build_prompt_graph_from_llm(llm_rel_dict)
        Gp = self.build_prompt_graph_from_llm_filtered(llm_rel_dict, self.args.relation2id)
        relation_graph2 = self.build_pyg_graph_from_triples(Gp["T"], Gp["R"])

        for e in range(self.args.metatrain_num_epoch):
            for batch in self.train_subgraph_dataloader:
                batch_loss = 0
                g1 = self.build_dgl_graph(self.args.ee_edges,self.args,'train')
                g2 = self.build_dgl_graph(self.args.ev_edges,self.args,'train')
                g3 = self.build_dgl_graph(self.args.vv_edges,self.args,'train')


                for g in (g1, g2, g3):
                    attach_and_compute_fuzziness(
                        g, self.fuzzy_modules,
                        node_time_key="time",
                        attr_edge_keys=("impact_km", "severity"),
                        attr_node_keys=("impact_km", "severity"),
                        rel_type_key="rel_type",
                        global_num_rel=self.global_R,  # ★ 通过参数传进来，而不是 self
                        use_mix=True,
                        tnorm="product",
                    )

                self.gloal_embeddings = self.multiviewRGCN(
                    graphs=[g1, g2, g3],
                    rel_types=[
                        g1.edata['rel_type'],
                        g2.edata['rel_type'],
                        g3.edata['rel_type']
                    ],
                    edge_weights=[g1.edata['weight'],
                        g2.edata['weight'],
                        g3.edata['weight']
                    ]
                )


                support_triplets = torch.cat([d[0] for d in batch])  # (h, r, t)
                # # 3️⃣ **生成关系-关系三元组**
                # generate_relation_triplets_llm
                relation_triplets = self.generate_relation_triplets_llm(
                    support_triplets.cpu().numpy(),
                    self.args.num_ent,
                    self.args.num_rel,
                    self.args.B,
                    relation_graph2
                )

                relation_triplets = torch.tensor(relation_triplets, dtype=torch.long, device=self.args.gpu)

                # 4️⃣ **计算关系嵌入**
                rel_emb = self.get_relation_emb(relation_triplets)

                # 5️⃣ **计算实体嵌入**
                batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch]).to(self.args.gpu)
                # # 获取所有唯一的节点


                self.get_ent_emb(batch_sup_g)
                sup_g_list = dgl.unbatch(batch_sup_g)

                for batch_i, data in enumerate(batch):

                    local2global,que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[1:]]
                    local_ent_emb = self.gloal_embeddings[local2global].to(self.args.gpu)


                    self.explainer.set_context(sup_g_list[batch_i])  # <-- 新增


                    ent_emb = sup_g_list[batch_i].ndata['h']


                    ent_emb_fused = self.GatedFusion(ent_emb, local_ent_emb)


                    loss = self.get_loss(que_tri, que_neg_tail_ent, que_neg_head_ent, ent_emb_fused,rel_emb)#负采样也不行

                    batch_loss += loss

                batch_loss /= len(batch)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                step += 1
                self.logger.info('step: {} | loss: {:.4f}'.format(step, batch_loss.item()))
                self.write_training_loss(batch_loss.item(), step)

                if step % self.args.metatrain_check_per_step == 0:
                    eval_res = self.evaluate_valid_subgraphs()
                    self.write_evaluation_result(eval_res, step)

                    # 取 overall 的 MRR 作为早停指标
                    current_mrr = eval_res['overall']['mrr']

                    if current_mrr > best_eval_rst['overall']['mrr']:
                        best_eval_rst = eval_res
                        best_step = step
                        self.logger.info('best model | overall mrr {:.4f}'.format(current_mrr))
                        self.save_checkpoint(step)
                        bad_count = 0

                        self.logger.info('new test')
                        self.logger.info('save best model')
                        self.save_model(best_step)

                        self.before_test_load()
                        self.evaluate_indtest_test_triples(num_cand=50)
                    else:
                        bad_count += 1
                        self.logger.info(
                            'best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                                best_step, best_eval_rst['overall']['mrr'], bad_count
                            )
                        )


        self.logger.info('finish meta-training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info("Training finished. Saving best model at step {}.".format(best_step))

        # Overall
        self.logger.info("[Overall]     MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            best_eval_rst['overall']['mrr'],
            best_eval_rst['overall']['hits@1'],
            best_eval_rst['overall']['hits@5'],
            best_eval_rst['overall']['hits@10']
        ))

        # Event Query
        self.logger.info("[Event Query] MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            best_eval_rst['event_query']['mrr'],
            best_eval_rst['event_query']['hits@1'],
            best_eval_rst['event_query']['hits@5'],
            best_eval_rst['event_query']['hits@10']
        ))

        # Entity Query
        self.logger.info("[Entity Query]MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            best_eval_rst['entity_query']['mrr'],
            best_eval_rst['entity_query']['hits@1'],
            best_eval_rst['entity_query']['hits@5'],
            best_eval_rst['entity_query']['hits@10']
        ))
        with open('./data/'+self.args.data_name+'/best_eval_result.json', "w") as f:
            json.dump(best_eval_rst, f, indent=4)
        #这里
        self.before_test_load()
        self.evaluate_indtest_test_triples(num_cand=50)

    def evaluate_valid_subgraphs(self):
        # from collections import defaultdict as ddict

        all_results = {
            "overall": ddict(float),
            "event_query": ddict(float),
            "entity_query": ddict(float)
        }

        for batch in self.valid_subgraph_dataloader:
            # Step 1: 支持集三元组
            support_triplets = torch.cat([d[0] for d in batch])

            # Step 2: 关系嵌入
            relation_triplets = self.generate_relation_triplets(
                support_triplets.cpu().numpy(),
                self.args.num_ent,
                self.args.num_rel,
                self.args.B
            )
            relation_triplets = torch.tensor(relation_triplets).to(self.args.gpu)
            rel_emb = self.get_relation_emb(relation_triplets)

            # Step 3: 实体嵌入（GNN + Gated Fusion）
            batch_sup_g = dgl.batch([get_g_bidir(d[0], self.args) for d in batch]).to(self.args.gpu)
            self.get_ent_emb(batch_sup_g)
            sup_g_list = dgl.unbatch(batch_sup_g)

            for batch_i, data in enumerate(batch):
                local2global = data[1]
                inv_ent_reidx = data[2]
                que_dataloader = data[3]

                # Step 4: 获取本子图上的融合嵌入
                ent_emb = sup_g_list[batch_i].ndata['h']
                local_ent_emb = self.gloal_embeddings[local2global].to(self.args.gpu)
                ent_emb_fused = self.GatedFusion(ent_emb, local_ent_emb)

                # Step 5: 评估 → 返回分类型指标
                subgraph_results = self.evaluate(
                    ent_emb_fused,
                    rel_emb,
                    que_dataloader,
                    inv_ent_reidx  # 即 local2global
                )

                # Step 6: 累加各类指标
                for category in ["overall", "event_query", "entity_query"]:
                    for k, v in subgraph_results[category].items():
                        all_results[category][k] += v

        # Step 7: 平均每类结果
        for category in all_results:
            for k in all_results[category]:
                all_results[category][k] /= self.args.num_valid_subgraph

        # Step 8: 打印日志
        self.logger.info('Valid on subgraphs')
        for cat in ["overall", "event_query", "entity_query"]:
            self.logger.info(f'[{cat}] MRR: {all_results[cat]["mrr"]:.4f}, '
                             f'H@1: {all_results[cat]["hits@1"]:.4f}, '
                             f'H@5: {all_results[cat]["hits@5"]:.4f}, '
                             f'H@10: {all_results[cat]["hits@10"]:.4f}')

        return all_results
