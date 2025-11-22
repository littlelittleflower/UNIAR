import torch
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv

class SingleViewRGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels):
        super().__init__()
        self.layer1 = RelGraphConv(in_dim, h_dim, num_rels, "basis", num_bases=4, activation=nn.ReLU())
        self.layer2 = RelGraphConv(h_dim, out_dim, num_rels, "basis", num_bases=4)

    def forward(self, g, feats, rel_types):
        h = self.layer1(g, feats, rel_types)
        h = self.layer2(g, h, rel_types)
        return h

class MultiViewRGCN(nn.Module):
    def __init__(self, num_nodes,in_dim,h_dim,out_dim,rel_nums):
        super().__init__()
        self.node_features = nn.Parameter(torch.randn(num_nodes, in_dim))  # 可训练嵌入
        self.view1 = SingleViewRGCN(in_dim, h_dim, out_dim, rel_nums[0])  # event-event
        self.view2 = SingleViewRGCN(in_dim, h_dim, out_dim, rel_nums[1])  # event-entity
        self.view3 = SingleViewRGCN(in_dim, h_dim, out_dim, rel_nums[2])  # entity-entity
        self.fusion = nn.Linear(out_dim * 3, out_dim)

    def forward(self, graphs, rel_types):
        feats = self.node_features.to('cuda:0')#?
        graphs = [graph.to('cuda:0') for graph in graphs]
        rel_types = [rel_type.to('cuda:0') for rel_type in rel_types]

        h1 = self.view1(graphs[0], feats, rel_types[0])
        h2 = self.view2(graphs[1], feats, rel_types[1])
        h3 = self.view3(graphs[2], feats, rel_types[2])
        h_cat = torch.cat([h1, h2, h3], dim=-1)
        return self.fusion(h_cat)
