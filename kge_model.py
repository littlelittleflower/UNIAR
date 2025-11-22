import torch.nn as nn
import torch


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.model_name = args.kge
        self.nrelation = args.num_rel
        self.emb_dim = args.emb_dim
        self.epsilon = 2.0

        self.gamma = torch.Tensor([args.gamma])
        self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.emb_dim])



        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE']:
            raise ValueError('model %s not supported' % self.model_name)

    def forward(self, sample, ent_emb, rel_emb=None, mode='single'):
        """
        KGE 计算三元组 (h, r, t) 的得分
        现在支持使用外部 `rel_emb`，如果不提供则默认使用 `self.relation_embedding`

        参数：
        - sample: (h, r, t) 三元组
        - ent_emb: 计算后的实体嵌入
        - rel_emb: 计算后的关系嵌入（如果不提供，则使用 `self.relation_embedding`）
        - mode: 计算方式（single, head-batch, tail-batch, rel-batch）

        返回：
        - 计算出的得分
        """
        self.entity_embedding = ent_emb
        self.relation_embedding = rel_emb
        # **如果 `ent_emb` 变成 (125, 32)，那么 sample 里的索引可能超出范围**


        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            if isinstance(sample, tuple):
                sample = sample[0]  # 只取第一个部分

                # **创建 `entity_mapping`**
            entity_mapping = {int(eid): i for i, eid in enumerate(torch.unique(sample[:, [0, 2]]).tolist())}


            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

            # else:
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            if isinstance(tail_part, tuple):
                tail_part = tail_part[0]
            if isinstance(head_part, tuple):
                head_part = head_part[0]
            # batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if head_part != None:
                try:
                    batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

                except IndexError:
                    print(head_part)



            if head_part == None:
                head = self.entity_embedding.unsqueeze(0)
            else:

                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1).long()
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)



        elif mode == 'tail-batch':
            head_part, tail_part = sample

            if tail_part != None:
                try:
                    batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

                except IndexError:
                    print(tail_part)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            if tail_part == None:
                tail = self.entity_embedding.unsqueeze(0)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1).long()#展开
                ).view(batch_size, negative_sample_size, -1)
                # print('123')
        elif mode == 'rel-batch':
            head_part, tail_part = sample

            if tail_part != None:
                batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 2]
            ).unsqueeze(1)

            if tail_part == None:
                relation = self.relation_embedding.unsqueeze(0)
            else:
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)

        # 选择 KGE 评分函数
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        # head = head.cpu()
        # relation = relation.cpu()
        # tail = tail.cpu()
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_phase, im_phase = torch.chunk(phase_relation, 2, dim=2)

        re_relation = torch.cos(re_phase)
        im_relation = torch.sin(re_phase)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score

