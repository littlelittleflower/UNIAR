import torch
from torch import optim
import numpy as np
import igraph
import dgl
import os
from utils import get_posttrain_train_valid_dataset
from torch.utils.data import DataLoader
from datasets import KGETrainDataset, KGEEvalDataset
from trainer import Trainer
import math


class PostTrainer(Trainer):
    def __init__(self, args):
        super(PostTrainer, self).__init__(args)
        self.args = args
        self.load_metatrain()#先预训练，再微调,得到训练好的参数
        self.test_model = None

        # dataloader
        train_dataset, valid_dataset = get_posttrain_train_valid_dataset(args)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.posttrain_bs,
                                      collate_fn=KGETrainDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.indtest_eval_bs,
                                      collate_fn=KGEEvalDataset.collate_fn)

        self.optimizer = optim.Adam(list(self.ent_init.parameters()) + list(self.rgcn.parameters())
                                    + list(self.kge_model.parameters()), lr=self.args.posttrain_lr)

    def load_metatrain(self):
        # state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])
        self.GatedFusion.load_state_dict(state['GatedFusion'])

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        sup_g_bidir.ndata['h'] = sup_g_bidir.ndata['feat']
        ent_emb = self.rgcn(sup_g_bidir)

        return ent_emb


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

    def get_relation_triplets(self, G_rel, B):
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

    def get_node_id(self,node, entity2id, event2id, offset):
        if self.is_event(node,event2id):
            return event2id[node] + offset
        else:
            return entity2id[node]
    def is_event(self,node_str,event2id):
        """判断是否为事件节点"""
        return node_str in event2id

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

        # 就是单向的
        g_rel = dgl.graph((src, dst), num_nodes=self.args.num_rel * 2).to(self.args.gpu)

        # 初始化节点特征，注意这里使用 self.args.rel_dim 作为特征维度（可根据你的设置调整）
        num_nodes = g_rel.num_nodes()
        # 这里初始化为全零，也可以使用随机初始化
        g_rel.ndata['feat'] = torch.zeros((num_nodes, self.args.rel_dim), device=self.args.gpu)

        g_rel.edata['type'] = relation_triplets[:, 2]

        g_rel.ndata['h'] = torch.randn(num_nodes, self.args.ent_dim).to(self.args.gpu)
        rel_emb = self.rgcn(g_rel)
        return rel_emb

    def build_dgl_graph(self,edge_list,args,label='test'):
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
        rel_map = {}
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
            fuzziness_list.append(fu_dict.get((h_id, relation2id[r], t_id)))
            rel_ids.append(rel_map[r])

        g = dgl.graph((src_ids, dst_ids), num_nodes=len(entity2id) + len(event2id))
        g.edata['rel_type'] = torch.tensor(rel_ids)
        g.edata['weight'] = torch.tensor(fuzziness_list)
        return g

    def train(self):
        self.logger.info('start fine-tuning')

        # print epoch test rst
        self.evaluate_indtest_test_triples(num_cand=50)

        for i in range(1, self.args.posttrain_num_epoch + 1):
            losses = []
            for batch in self.train_dataloader:
                support_triplets = batch[0]
                relation_triplets = self.generate_relation_triplets(
                    support_triplets.cpu().numpy(),
                    self.args.num_ent,
                    self.args.num_rel,
                    self.args.B
                )
                relation_triplets = torch.tensor(relation_triplets).to(self.args.gpu)
                rel_emb = self.get_relation_emb(relation_triplets)



                pos_triple, neg_tail_ent, neg_head_ent = [b.to(self.args.gpu) for b in batch]

                ent_emb = self.get_ent_emb(self.indtest_train_g)
                test_emb = self.get_test_embedd()

                ent_emb_fused = self.GatedFusion(ent_emb, test_emb)
                loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb_fused, rel_emb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            self.logger.info('epoch: {} | loss: {:.4f}'.format(i, np.mean(losses)))

            if i % self.args.posttrain_check_per_epoch == 0:
                self.evaluate_indtest_test_triples(num_cand=50)

    def evaluate_indtest_valid_triples(self, num_cand='all'):
        ent_emb = self.get_ent_emb(self.indtest_train_g)

        results = self.evaluate(ent_emb, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
