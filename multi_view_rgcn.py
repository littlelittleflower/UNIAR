import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn

class WeightedRelGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=4):
        super().__init__()
        self.num_rels = num_rels
        self.rel_emb = nn.Parameter(torch.Tensor(num_rels, in_dim, out_dim))
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, g, feat, rel_type, edge_weight=None):
        with g.local_scope():
            # 获取源节点特征
            src_feat = feat[g.edges()[0]]  # [E, in_dim]

            # 按照边的关系类型查对应的权重矩阵
            w = self.rel_emb[rel_type].to(src_feat.dtype) # [E, in_dim, out_dim]

            # 消息 = h_src @ W_r
            msg = torch.bmm(src_feat.unsqueeze(1), w).squeeze(1)  # [E, out_dim]

            if edge_weight is not None:
                msg = msg * edge_weight.unsqueeze(-1)  # 加权消息传播

            g.edata['msg'] = msg
            g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'h'))

            return g.ndata['h']

class GCNView(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels, num_bases=4):
        super().__init__()
        self.layer1 = WeightedRelGraphConv(in_dim, h_dim, num_rels, num_bases)
        self.layer2 = WeightedRelGraphConv(h_dim, out_dim, num_rels, num_bases)

    def forward(self, g, feat, rel_type, edge_weight=None):
        h = self.layer1(g, feat, rel_type, edge_weight=edge_weight)
        h = torch.relu(h)
        h = self.layer2(g, h, rel_type, edge_weight=edge_weight)
        return h


class MultiViewRGCN(nn.Module):
    def __init__(self, num_nodes, in_dim, h_dim, out_dim, rel_nums,args):
        super().__init__()
        self.node_features = nn.Parameter(torch.randn(num_nodes, args.in_dim))

        # 三个视图子网络（EE / EV / VV）
        self.view1 = GCNView(in_dim, h_dim, out_dim, rel_nums[0])
        self.view2 = GCNView(in_dim, h_dim, out_dim, rel_nums[1])
        self.view3 = GCNView(in_dim, h_dim, out_dim, rel_nums[2])

        # 融合 attention 层（将 [v1, v2, v3] 映射成权重）
        self.att_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 1)
        )

    def forward(self, graphs, rel_types, edge_weights=None):
        """
        参数：
            graphs: [g1, g2, g3] 三个视图的 DGLGraph
            rel_types: [r1, r2, r3] 每条边的关系类型
            edge_weights: [w1, w2, w3] 每条边的三元组模糊度 fu（可选）
        """
        device = 'cuda:0'
        dtype = self.node_features.dtype

        # 将 edge_weight 拆出来
        ew1 = edge_weights[0].to(device) if edge_weights else None
        ew2 = edge_weights[1].to(device) if edge_weights else None
        ew3 = edge_weights[2].to(device) if edge_weights else None

        # 三个视图分别 message passing
        emb1 = self.view1(graphs[0].to(device), self.node_features, rel_types[0].to(device), edge_weight=ew1)
        emb2 = self.view2(graphs[1].to(device), self.node_features, rel_types[1].to(device), edge_weight=ew2)
        emb3 = self.view3(graphs[2].to(device), self.node_features, rel_types[2].to(device), edge_weight=ew3)

        # 拼接 -> attention 融合
        stacked = torch.stack([emb1, emb2, emb3], dim=1).to(dtype) # [N, 3, D]
        attn_scores = self.att_mlp(stacked)  # [N, 3, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [N, 3, 1]
        fused = torch.sum(attn_weights * stacked, dim=1)  # [N, D]

        return fused  # 返回每个节点最终表示
