from torch.utils.tensorboard import SummaryWriter
import os
import json
from fusion_model import GatedFusion
from utils import Log
from torch.utils.data import DataLoader
from ent_init_model import EntInit
from rgcn_model import RGCN
from multi_view_rgcn import MultiViewRGCN
from kge_model import KGEModel
import torch
import torch.nn.functional as F
from collections import defaultdict as ddict
from utils import get_indtest_test_dataset_and_train_g
from datasets import KGEEvalDataset


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.name = args.name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()



        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        indtest_test_dataset, indtest_train_g = get_indtest_test_dataset_and_train_g(args)
        self.indtest_train_g = indtest_train_g.to(args.gpu)
        self.indtest_test_dataloader = DataLoader(indtest_test_dataset, batch_size=args.indtest_eval_bs,
                                                  shuffle=False, collate_fn=KGEEvalDataset.collate_fn)

        # models
        self.ent_init = EntInit(args).to(args.gpu)
        self.GatedFusion = GatedFusion(args).to(args.gpu)
        self.multiviewRGCN = MultiViewRGCN(args.num_nodes,args.in_dim,args.h_dim,args.out_dim,args.rel_nums,args).to(args.gpu)
        self.rgcn = RGCN(args).to(args.gpu)
        self.kge_model = KGEModel(args).to(args.gpu)

    def save_checkpoint(self, step):
        # 保存需要的参数
        state = {'ent_init': self.ent_init.state_dict(),
                 'rgcn': self.rgcn.state_dict(),
                 'kge_model': self.kge_model.state_dict(),
                 'GatedFusion':self.GatedFusion.state_dict(),
                 'multiviewRGCN':self.multiviewRGCN.state_dict()
                 }
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_evaluation_result(self, results, step):
        """
        Write evaluation results (overall, event_query, entity_query) to TensorBoard.
        """
        for group_name, metrics in results.items():  # group_name: 'overall', 'event_query', etc.
            for metric_name, value in metrics.items():  # metric_name: 'mrr', 'hits@10', etc.
                tag = f"evaluation/{group_name}/{metric_name}"
                self.writer.add_scalar(tag, value, step)


    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])
        self.GatedFusion.load_state_dict(state['GatedFusion'])
        # self.multiviewRGCN.load_state_dict(state['multiviewRGCN'])#?

    def check_dim_consistency(self,ent_emb, rel_emb, model_name=""):
        """
        检查实体/关系嵌入是否维度一致，适用于 ComplEx、RotatE 等模型。

        参数:
            ent_emb: 实体嵌入 Tensor [N, d]
            rel_emb: 关系嵌入 Tensor [R, d]
            model_name: 模型名称，用于模型特定限制

        抛出:
            ValueError：如果维度不一致，或不满足模型要求
        """
        if ent_emb is None or rel_emb is None:
            raise ValueError("实体或关系嵌入为 None")

        if ent_emb.size(-1) != rel_emb.size(-1):
            raise ValueError(f" 维度不一致：ent_emb={ent_emb.size(-1)}, rel_emb={rel_emb.size(-1)}")

        if model_name in ['ComplEx', 'RotatE'] and ent_emb.size(-1) % 2 != 0:
            raise ValueError(f" {model_name} 要求 embedding dim 是偶数，当前为 {ent_emb.size(-1)}")


    def get_loss(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb,
                 explain_K=20, select_m=3,  # 候选路径数量 & 选择数量
                 tnorm="product",
                 lambda_abd=1.0, lambda_cons=0.5, lambda_len=0.01, margin=0.2,
                 temp=1.0):
        """
        在原 KGE 损失上，联合溯因解释的三项损失：
          - L_abd: 解释排序/选择（好路径>坏路径）
          - L_cons: 解释→预测一致性
          - L_len: 简洁性（惩罚过长路径）
        其余参数与你现有训练保持兼容。
        """

        # -----------------------------
        # 0) 原有 KGE 损失（保留不动）
        # -----------------------------
        if neg_tail_ent is None:
            raise ValueError("tail_part is None! Check data loader or negative sampling.")

        self.check_dim_consistency(ent_emb, rel_emb, model_name=self.args.kge)

        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score], dim=0)
        neg_score = F.softmax(neg_score * self.args.adv_temp, dim=1).detach() * F.logsigmoid(-neg_score)

        pos_score_raw = self.kge_model(tri, ent_emb, rel_emb)  # 未过sigmoid的原始打分
        pos_score = F.logsigmoid(pos_score_raw).squeeze(dim=1)  # 用于原始loss
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        kge_loss = (positive_sample_loss + negative_sample_loss) / 2

        # -----------------------------
        # 1) 生成候选解释路径（ADD）
        #    假定你有 self.explainer.generate(tri, K) -> list[list[Path]]
        #    每条 Path 含若干 Edge，每个 Edge 带 mu_rel/mu_time/mu_attr 与 length=1
        # -----------------------------
        # paths_batch: 长度 = batch_size，每个元素是该样本的候选路径列表（长度<=explain_K）
        paths_batch = self.explainer.generate(tri, K=explain_K)  # 你来提供 explainer（见下文“你需要提供的代码”）

        # -----------------------------
        # 2) 路径打分 + Gumbel/soft 选择（ADD）
        #    S(P)= t-norm(mu_rel,mu_time,mu_attr) 聚合 - 长度惩罚
        # -----------------------------
        def tnorm_agg(vals, kind="product"):
            if kind == "min":
                return torch.min(vals)
            elif kind == "lukasiewicz":
                # 逐个折叠 a⊗_L b = max(0, a + b - 1)
                s = vals[0]
                for v in vals[1:]:
                    s = torch.clamp(s + v - 1.0, min=0.0, max=1.0)
                return s
            else:  # product
                return torch.prod(vals)

        # per-sample结果容器
        S_list = []  # 每个样本候选路径的分数 [num_paths]
        L_list = []  # 对应路径长度
        Pi_list = []  # soft选择权重 [num_paths]（top-m 近似）
        top_paths_list = []  # 可选：存放被选择的前m条路径索引

        # 针对每个样本
        for sample_paths in paths_batch:
            if len(sample_paths) == 0:
                S_list.append(torch.zeros(1, device=pos_score_raw.device))
                L_list.append(torch.ones(1, device=pos_score_raw.device))
                Pi_list.append(torch.ones(1, device=pos_score_raw.device))  # 没有解释时给个恒等
                top_paths_list.append([])
                continue

            scores = []
            lens = []
            for P in sample_paths:
                # 把每条路径的各类型模糊度做聚合
                mu_rel = torch.tensor([e.mu_rel for e in P.edges], device=pos_score_raw.device).clamp(0, 1)
                mu_time = torch.tensor([e.mu_time for e in P.edges], device=pos_score_raw.device).clamp(0, 1)
                mu_attr = torch.tensor([e.mu_attr for e in P.edges], device=pos_score_raw.device).clamp(0, 1)

                rel = tnorm_agg(mu_rel, tnorm)
                tim = tnorm_agg(mu_time, tnorm)
                att = tnorm_agg(mu_attr, tnorm)
                mix = tnorm_agg(torch.stack([rel, tim, att]), tnorm)

                # 路径长度
                Lp = torch.tensor(len(P.edges), dtype=torch.float32, device=pos_score_raw.device)

                # 路径总分：mix - λ_len * length   （排序时不加 λ_len，训练时会用 L_len 统一惩罚也可）
                S_p = mix
                scores.append(S_p)
                lens.append(Lp)

            scores = torch.stack(scores)  # [num_paths]
            lens = torch.stack(lens)  # [num_paths]

            # soft/top-m 选择：用softmax温度做近似（如需严格TopK，可用Gumbel-TopK）
            pi = torch.softmax(scores / max(1e-6, temp), dim=0)

            S_list.append(scores)
            L_list.append(lens)
            Pi_list.append(pi)

            # 可选：记录top-m索引用于可视化
            if select_m > 0:
                top_idx = torch.topk(scores, k=min(select_m, scores.numel())).indices.tolist()
                top_paths_list.append(top_idx)
            else:
                top_paths_list.append([])

        # -----------------------------
        # 3) L_abd：解释排序/选择（ADD）
        #    正路径：候选中分数高 & 时间/关系一致；负路径：扰动（反向时间/换关系/断边）
        #    这里给一个简化的 margin ranking 实现：正=soft选择的期望，负=从自造负路径采样
        # -----------------------------
        # 需要 explainer 提供负路径生成（或在 explainer 内部做），这里按接口调用
        neg_paths_batch = self.explainer.corrupt(paths_batch)  # 与 paths_batch 结构一致的负候选

        def batch_paths_score(paths_batch):
            """和上面一样的打分，封成函数用于负样本"""
            #这里应该就是三类模糊度兜底吧，没有做区分
            out = []
            for sample_paths in paths_batch:
                if len(sample_paths) == 0:
                    out.append(torch.zeros(1, device=pos_score_raw.device))
                    continue
                s = []
                for P in sample_paths:
                    mu_rel = torch.tensor([e.mu_rel for e in P.edges], device=pos_score_raw.device).clamp(0, 1)
                    mu_time = torch.tensor([e.mu_time for e in P.edges], device=pos_score_raw.device).clamp(0, 1)
                    mu_attr = torch.tensor([e.mu_attr for e in P.edges], device=pos_score_raw.device).clamp(0, 1)
                    rel = tnorm_agg(mu_rel, tnorm)
                    tim = tnorm_agg(mu_time, tnorm)
                    att = tnorm_agg(mu_attr, tnorm)
                    mix = tnorm_agg(torch.stack([rel, tim, att]), tnorm)
                    s.append(mix)
                out.append(torch.stack(s))
            return out

        pos_paths_scores = S_list
        neg_paths_scores = batch_paths_score(neg_paths_batch)

        # 将每个样本的解释分做“soft选择的期望”
        pos_expect = []
        neg_expect = []
        for pos_s, pi, neg_s in zip(pos_paths_scores, Pi_list, neg_paths_scores):
            pos_expect.append((pi * pos_s).sum())
            # 负样本：取最大或平均，这里取最大（更强对比）
            if neg_s.numel() > 0:
                neg_expect.append(torch.max(neg_s))
            else:
                neg_expect.append(torch.tensor(0.0, device=pos_score_raw.device))
        pos_expect = torch.stack(pos_expect)  # [batch]
        neg_expect = torch.stack(neg_expect)  # [batch]

        L_abd = torch.clamp(margin - (pos_expect - neg_expect), min=0.0).mean()

        # -----------------------------
        # 4) L_cons：解释→预测一致性（ADD）
        #    让解释强度（期望分）与主预测概率一致：ReLU(S_bar - y_hat)
        # -----------------------------
        y_hat = torch.sigmoid(pos_score_raw).squeeze(dim=1)  # [batch]
        S_bar = pos_expect.clamp(0, 1)  # [batch]
        L_cons = torch.relu(S_bar - y_hat).mean()

        # -----------------------------
        # 5) L_len：简洁性正则（ADD）
        #    用被soft选择的权重对长度求期望，再做平均
        # -----------------------------
        len_expect = []
        for lens, pi in zip(L_list, Pi_list):
            len_expect.append((pi * lens).sum())
        len_expect = torch.stack(len_expect)
        L_len = len_expect.mean()

        # -----------------------------
        # 6) 总损失
        # -----------------------------
        loss = kge_loss + lambda_abd * L_abd + lambda_cons * L_cons + lambda_len * L_len
        return loss

    def get_loss_1(self, tri, neg_tail_ent, neg_head_ent, ent_emb, rel_emb):
        """
        计算 KGE 损失，同时利用增强后的 `rel_emb`

        参数：
        - tri: 真实三元组 (h, r, t)
        - neg_tail_ent: 负样本 (h, r, t')
        - neg_head_ent: 负样本 (h', r, t)
        - ent_emb: 增强后的实体嵌入
        - rel_emb: 通过 GNN 计算的增强关系嵌入

        返回：
        - loss: KGE 训练损失
        """

        if neg_tail_ent is None:
            raise ValueError("tail_part is None! Check data loader or negative sampling.")

        # 计算负样本得分
        # tail-batch出错
        self.check_dim_consistency(ent_emb, rel_emb, model_name=self.args.kge)
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, rel_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, rel_emb, mode='head-batch')

        neg_score = torch.cat([neg_tail_score, neg_head_score], dim=0)  # 拼接负样本
        # neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
        #              * F.logsigmoid(-neg_score)).sum(dim=1)

        neg_score = F.softmax(neg_score * self.args.adv_temp, dim=1).detach() * F.logsigmoid(-neg_score)

        # 计算正样本得分
        pos_score = self.kge_model(tri, ent_emb, rel_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        # 计算损失
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        sup_g_bidir.ndata['h'] = sup_g_bidir.ndata['feat']
        ent_emb = self.rgcn(sup_g_bidir)

        return ent_emb

    def is_event3(self, global_id):
        return global_id >= len(self.args.test_entity2id)  # 如果test文件夹的偏移区分

    def is_event2(self, global_id):
        return global_id >= len(self.args.entity2id)  # 如果用偏移区分

    def evaluate(self, ent_emb, rel_emb, eval_dataloader, local2global=None, num_cand='all'):
        results = ddict(float)
        event_results = ddict(float)
        entity_results = ddict(float)
        count = 0
        event_count = 0
        entity_count = 0

        eval_dataloader.dataset.num_cand = num_cand

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                if rel_emb is None:
                    relation_triplets = self.generate_relation_triplets(
                        pos_triple.cpu().numpy(),
                        self.args.num_ent,
                        self.args.num_rel,
                        self.args.B
                    )
                    relation_triplets = torch.tensor(relation_triplets).to(self.args.gpu)
                    rel_emb = self.get_relation_emb(relation_triplets)

                b_range = torch.arange(pos_triple.size(0), device=self.args.gpu)

                # ----- tail prediction -----
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, mode='tail-batch')
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.bool(), -torch.ones_like(pred) * 1e7, pred)
                pred[b_range, tail_idx] = target_pred
                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, tail_idx]

                # ----- head prediction -----
                pred = self.kge_model((pos_triple, None), ent_emb, rel_emb, mode='head-batch')
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.bool(), -torch.ones_like(pred) * 1e7, pred)
                pred[b_range, head_idx] = target_pred
                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, head_idx]

                # ----- 分开统计 -----
                for i in range(pos_triple.size(0)):
                    # tail 评估
                    tail_local_id = tail_idx[i].item()
                    tail_global_id = local2global[tail_local_id] if local2global else tail_local_id
                    is_event = self.is_event2(tail_global_id)

                    rank = tail_ranks[i].item()
                    count += 1
                    results['mr'] += rank
                    results['mrr'] += 1.0 / (rank + 1e-8)
                    for k in [1, 5, 10]:
                        if rank <= k:
                            results[f'hits@{k}'] += 1

                    if is_event:
                        event_count += 1
                        event_results['mr'] += rank
                        event_results['mrr'] += 1.0 / (rank + 1e-8)
                        for k in [1, 5, 10]:
                            if rank <= k:
                                event_results[f'hits@{k}'] += 1
                    else:
                        entity_count += 1
                        entity_results['mr'] += rank
                        entity_results['mrr'] += 1.0 / (rank + 1e-8)
                        for k in [1, 5, 10]:
                            if rank <= k:
                                entity_results[f'hits@{k}'] += 1

                    # head 评估
                    head_local_id = head_idx[i].item()
                    head_global_id = local2global[head_local_id] if local2global else head_local_id
                    is_event = self.is_event2(head_global_id)

                    rank = head_ranks[i].item()
                    count += 1
                    results['mr'] += rank
                    results['mrr'] += 1.0 / (rank + 1e-8)
                    for k in [1, 5, 10]:
                        if rank <= k:
                            results[f'hits@{k}'] += 1

                    if is_event:
                        event_count += 1
                        event_results['mr'] += rank
                        event_results['mrr'] += 1.0 / (rank + 1e-8)
                        for k in [1, 5, 10]:
                            if rank <= k:
                                event_results[f'hits@{k}'] += 1
                    else:
                        entity_count += 1
                        entity_results['mr'] += rank
                        entity_results['mrr'] += 1.0 / (rank + 1e-8)
                        for k in [1, 5, 10]:
                            if rank <= k:
                                entity_results[f'hits@{k}'] += 1


        else:
            for _ in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]
                    b_range = torch.arange(pos_triple.size(0), device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size(0), dtype=torch.long, device=self.args.gpu)

                    if rel_emb is None:
                        relation_triplets = self.generate_relation_triplets(
                            pos_triple.cpu().numpy(),
                            self.args.num_ent,
                            self.args.num_rel,
                            self.args.B
                        )
                        relation_triplets = torch.tensor(relation_triplets).to(self.args.gpu)
                        rel_emb = self.get_relation_emb(relation_triplets)

                    # ---- tail-batch ----
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, rel_emb, mode='tail-batch')
                    tail_ranks = 1 + \
                                 torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                                     b_range, target_idx]

                    # ---- head-batch ----
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, rel_emb, mode='head-batch')
                    head_ranks = 1 + \
                                 torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                                     b_range, target_idx]

                    # ---- 统计 ----
                    for i in range(pos_triple.size(0)):
                        for role, idx, rank in [('tail', pos_triple[i][2], tail_ranks[i]),
                                                ('head', pos_triple[i][0], head_ranks[i])]:
                            local_id = idx.item()
                            global_id = local2global[local_id] if local2global else local_id
                            is_event = self.is_event3(global_id)

                            rank = rank.item()
                            count += 1
                            results['mr'] += rank
                            results['mrr'] += 1.0 / (rank + 1e-8)
                            for k in [1, 5, 10]:
                                if rank <= k:
                                    results[f'hits@{k}'] += 1

                            if is_event:
                                event_count += 1
                                event_results['mr'] += rank
                                event_results['mrr'] += 1.0 / (rank + 1e-8)
                                for k in [1, 5, 10]:
                                    if rank <= k:
                                        event_results[f'hits@{k}'] += 1
                            else:
                                entity_count += 1
                                entity_results['mr'] += rank
                                entity_results['mrr'] += 1.0 / (rank + 1e-8)
                                for k in [1, 5, 10]:
                                    if rank <= k:
                                        entity_results[f'hits@{k}'] += 1

        # ---- 结果归一化 ----
        for k in results:
            results[k] /= count
        for k in event_results:
            event_results[k] /= event_count
        for k in entity_results:
            entity_results[k] /= entity_count



        return {
            "overall": results,
            "event_query": event_results,
            "entity_query": entity_results
        }

    def load_multiview_model_for_inductive(self,test_model, checkpoint_path, device='cuda:0'):
        """
        从训练好的模型中加载 MultiViewRGCN 的可迁移参数到测试图模型中（归纳设置）。

        参数:
            test_model: 构建好的 MultiViewRGCN（针对测试图结构初始化）
            checkpoint_path: 训练阶段保存的模型路径
            device: 加载模型到哪个设备
        """
        import torch

        # 加载 checkpoint
        print(f"🔍 Loading pretrained MultiViewRGCN from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'multiviewRGCN' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'multiviewRGCN'!")

        pretrained_state = checkpoint['multiviewRGCN']
        current_state = test_model.state_dict()

        # 过滤匹配的参数（跳过形状不匹配的）
        filtered = {
            k: v for k, v in pretrained_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }

        # 加载
        missing_keys, unexpected_keys = test_model.load_state_dict(filtered, strict=False)

        print(f" Loaded {len(filtered)} matching parameters into test MultiViewRGCN.")
        if missing_keys:
            print(f" Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

        return test_model

    def get_test_embedd(self):
        import torch.nn as nn

        test_num_nodes = len(self.args.test_entity2id) + len(self.args.test_event2id)

        # 每次都构建图，但手动释放
        test_g1 = self.build_dgl_graph(self.args.test_ee_edges, self.args).to(self.args.gpu)
        test_g2 = self.build_dgl_graph(self.args.test_ev_edges, self.args).to(self.args.gpu)
        test_g3 = self.build_dgl_graph(self.args.test_vv_edges, self.args).to(self.args.gpu)

        #  模型只构建一次，后续复用（在类中先定义 self.test_model = None）
        if self.test_model is None:
            self.test_model = MultiViewRGCN(
                num_nodes=test_num_nodes,
                in_dim=self.args.in_dim,
                h_dim=self.args.h_dim,
                out_dim=self.args.out_dim,
                rel_nums=self.args.test_rel_nums,args=self.args).to(self.args.gpu)

            #  加载参数，只做一次
            self.load_multiview_model_for_inductive(
                self.test_model,
                checkpoint_path=os.path.join(self.state_path, self.name + '.best'),
                device=self.args.gpu
            )

        # 每次只更新节点嵌入（共享模型）
        self.test_model.node_features = nn.Parameter(
            torch.randn(test_num_nodes, self.args.in_dim).to(self.args.gpu)
        )

        # 前向传播
        with torch.no_grad():
            test_emb = self.test_model(
                graphs=[test_g1, test_g2, test_g3],
                rel_types=[
                    test_g1.edata['rel_type'],
                    test_g2.edata['rel_type'],
                    test_g3.edata['rel_type']
                ]
            ).detach()

        # 强制释放图显存（避免残留）
        del test_g1, test_g2, test_g3
        del self.test_model.node_features
        torch.cuda.empty_cache()

        return test_emb

    def evaluate_indtest_test_triples(self, num_cand='all'):
        """do evaluation on test triples of ind-test-graph"""
        ent_emb = self.get_ent_emb(self.indtest_train_g)  # indtest的dgl双向图
        test_emb = self.get_test_embedd()

        ent_emb_fused = self.GatedFusion(ent_emb, test_emb)

        results = self.evaluate(ent_emb_fused, None, self.indtest_test_dataloader,  num_cand=num_cand)

        self.logger.info(f'test on ind-test-graph, sample {num_cand}')
        # 输出 Overall 结果
        overall = results['overall']
        self.logger.info("[Overall]     MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            overall['mrr'], overall['hits@1'], overall['hits@5'], overall['hits@10']
        ))

        # 输出 Event Query 结果
        event = results['event_query']
        self.logger.info("[Event Query] MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            event['mrr'], event['hits@1'], event['hits@5'], event['hits@10']
        ))

        # 输出 Entity Query 结果
        entity = results['entity_query']
        self.logger.info("[Entity Query]MRR: {:.4f}, Hits@1: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}".format(
            entity['mrr'], entity['hits@1'], entity['hits@5'], entity['hits@10']
        ))


        del self.test_model
        self.test_model = None
        torch.cuda.empty_cache()

        return results
