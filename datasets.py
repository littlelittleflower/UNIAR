from torch.utils.data import Dataset, DataLoader
import lmdb
from utils import deserialize
import numpy as np
import torch

import numpy as np

def sample_hard_negatives(ent_emb, nentity, que_tri, hr2t, rt2h, num_neg=10):
    """
    生成 Hard Negatives：
    - 选择最接近正样本的负样本
    - 让模型更难学习
    """
    que_neg_tail_ent = []
    que_neg_head_ent = []

    entity_ids = np.arange(nentity)  # 所有实体索引
    ent_emb_np = ent_emb.cpu().detach().numpy()  # 转换为 NumPy

    for h, r, t in que_tri:
        # **候选负样本（排除正样本）**
        tail_candidates = np.delete(entity_ids, hr2t[(h, r)])
        head_candidates = np.delete(entity_ids, rt2h[(r, t)])

        # **计算负样本与正样本的相似性**
        tail_dists = np.linalg.norm(ent_emb_np[tail_candidates] - ent_emb_np[t], axis=1)
        head_dists = np.linalg.norm(ent_emb_np[head_candidates] - ent_emb_np[h], axis=1)

        # **选择 Hard Negatives**
        tail_sorted = tail_candidates[np.argsort(tail_dists)[:num_neg]]
        head_sorted = head_candidates[np.argsort(head_dists)[:num_neg]]

        que_neg_tail_ent.append(tail_sorted)
        que_neg_head_ent.append(head_sorted)

    return np.array(que_neg_tail_ent), np.array(que_neg_head_ent)

class TrainSubgraphDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=5, lock=False)
        # with self.env.begin() as txn:  # 开启事务
        #     cursor = txn.cursor()  # 获取游标
        #     for key, value in cursor:
        #         print(f"Key: {key.decode()}, Value: {value[:100]}")  # 只打印前 100 字节
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())
        # print('12')

    def __len__(self):
        return self.args.num_train_subgraph

    @staticmethod
    def collate_fn(data):
        return data

    def __getitem__(self, idx):
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, que_tri, hr2t, rt2h,inv_ent_reidx = deserialize(txn.get(str_id))
        nentity = len(np.unique(np.array(sup_tri)[:, [0, 2]]))

        que_neg_tail_ent = [np.random.choice(np.delete(np.arange(nentity), hr2t[(h, r)]),
                                        self.args.metatrain_num_neg) for h, r, t in que_tri]

        que_neg_head_ent = [np.random.choice(np.delete(np.arange(nentity), rt2h[(r, t)]),
                                        self.args.metatrain_num_neg) for h, r, t in que_tri]

        local2global = torch.tensor([inv_ent_reidx[i] for i in range(len(inv_ent_reidx))])

        return torch.tensor(sup_tri), local2global,torch.tensor(que_tri), \
               torch.tensor(que_neg_tail_ent), torch.tensor(que_neg_head_ent)


class ValidSubgraphDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=5, lock=False)
        self.subgraphs_db = self.env.open_db("valid_subgraphs".encode())

    def __len__(self):
        txn = self.env.begin(db=self.subgraphs_db)
        num = txn.stat()['entries']
        return num

    @staticmethod
    def collate_fn(data):
        return data

    def __getitem__(self, idx):
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, que_tri, hr2t, rt2h,inv_ent_reidx = deserialize(txn.get(str_id))#304,33?读取的是子图

        nentity = len(np.unique(np.array(sup_tri)[:, [0, 2]]))
        local2global = torch.tensor([inv_ent_reidx[i] for i in range(len(inv_ent_reidx))])

        que_dataset = KGEEvalDataset(self.args, que_tri, nentity, hr2t, rt2h)
        que_dataloader = DataLoader(que_dataset, batch_size=len(que_tri),
                                    collate_fn=KGEEvalDataset.collate_fn)

        return torch.tensor(sup_tri), local2global,inv_ent_reidx, que_dataloader



class KGETrainDataset(Dataset):
    def __init__(self, args, train_triples, num_ent, num_neg, hr2t, rt2h):
        self.args = args
        self.triples = train_triples
        self.num_ent = num_ent
        self.num_neg = num_neg
        self.hr2t = hr2t
        self.rt2h = rt2h

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple

        neg_tail_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                        self.num_neg)

        neg_head_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                        self.num_neg)

        pos_triple = torch.LongTensor(pos_triple)
        neg_tail_ent = torch.from_numpy(neg_tail_ent)
        neg_head_ent = torch.from_numpy(neg_head_ent)

        return pos_triple, neg_tail_ent, neg_head_ent

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        neg_tail_ent = torch.stack([_[1] for _ in data], dim=0)
        neg_head_ent = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, neg_tail_ent, neg_head_ent


class KGEEvalDataset(Dataset):
    def __init__(self, args, eval_triples, num_ent, hr2t, rt2h):
        self.args = args
        self.triples = eval_triples
        self.num_ent = num_ent
        self.hr2t = hr2t
        self.rt2h = rt2h
        self.num_cand = 'all'

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple
        if self.num_cand == 'all':
            tail_label, head_label = self.get_label(self.hr2t[(h, r)], self.rt2h[(r, t)])
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                             self.num_cand)

            neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                             self.num_cand)
            tail_cand = torch.from_numpy(np.concatenate(([t], neg_tail_cand)))
            head_cand = torch.from_numpy(np.concatenate(([h], neg_head_cand)))

            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_cand, head_cand

    def get_label(self, true_tail, true_head):
        y_tail = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_tail:
            y_tail[e] = 1.0
        y_head = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_head:
            y_head[e] = 1.0

        return torch.FloatTensor(y_tail), torch.FloatTensor(y_head)

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_label_or_cand = torch.stack([_[1] for _ in data], dim=0)
        head_label_or_cand = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, tail_label_or_cand, head_label_or_cand
