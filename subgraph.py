import pickle
import torch
import numpy as np
from collections import defaultdict as ddict
import lmdb
from tqdm import tqdm
import random
from utils import serialize, get_g, get_hr2t_rt2h_sup_que
import dgl
def check_event_ratio(inv_ent_reidx, event_ids, max_ratio=0.2, verbose=True):
    """
    检查 inv_ent_reidx 中事件节点的占比是否超过设定阈值 max_ratio。

    参数:
        inv_ent_reidx (dict): local_id -> global_id 映射
        event_ids (list or set): 所有事件节点的全局 ID
        max_ratio (float): 允许的最大事件节点占比
        verbose (bool): 是否打印日志

    返回:
        bool: True 表示通过检查，False 表示事件节点占比过高，建议重采样
    """
    event_ids_set = set(event_ids)
    total = len(inv_ent_reidx)
    event_count = sum(1 for gid in inv_ent_reidx.values() if gid in event_ids_set)

    ratio = event_count / total if total > 0 else 0

    if verbose:
        if ratio > max_ratio:
            print(f"[采样跳过] 事件节点占比过高：{ratio:.2%}（事件={event_count}，总={total}）")
        else:
            print(f"[采样成功] 节点总数={total}，事件={event_count}，事件占比={ratio:.2%}")

    return ratio <= max_ratio


def check_mapping_consistency(inv_ent_reidx, entity2id, event2id):
    """
    检查 inv_ent_reidx 中的 global ID 是否与 entity2id + event2id 构建的全局 ID 映射一致。
    输出实体节点、事件节点、以及不一致项。
    """
    entity_ids = set(entity2id.values())
    event_offset = len(entity2id)
    event_ids = set(range(event_offset, event_offset + len(event2id)))

    wrong_ids = []
    entity_count = 0
    event_count = 0

    for local_id, global_id in inv_ent_reidx.items():
        if global_id in entity_ids:
            entity_count += 1
        elif global_id in event_ids:
            event_count += 1
        else:
            wrong_ids.append((local_id, global_id))

    print("映射检查完成")
    print(f"实体节点数: {entity_count}")
    print(f"事件节点数: {event_count}")
    if wrong_ids:
        print(f"发现 {len(wrong_ids)} 个未在 entity2id 或 event2id 中的节点:")
        for local_id, global_id in wrong_ids:
            print(f"  local_id {local_id} → global_id {global_id}")
    else:
        print(" 所有节点映射一致！")

    return {
        "entity": entity_count,
        "event": event_count,
        "invalid": wrong_ids
    }



def gen_subgraph_datasets(args):
    print(f'-----There is no sub-graphs for {args.data_name}, so start generating sub-graphs before meta-training!-----')
    data = pickle.load(open(args.data_path, 'rb'))
    train_g = get_g(data['train_graph']['train'] + data['train_graph']['valid']
                    + data['train_graph']['test'])



    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, train_g) * 2
    map_size = (args.num_train_subgraph + args.num_valid_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=2)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())
    valid_subgraphs_db = env.open_db("valid_subgraphs".encode())

    for idx in tqdm(range(args.num_train_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g)
        with env.begin(write=True, db=train_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))

    for idx in tqdm(range(args.num_valid_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g)
        with env.begin(write=True, db=valid_subgraphs_db) as txn:
            txn.put(str_id, serialize(datum))


def sample_one_subgraph(args, bg_train_g):
    import random
    from collections import defaultdict as ddict
    import numpy as np
    import torch
    import dgl

    num_entities = len(args.entity2id)
    num_events = len(args.event2id)
    total_nodes = bg_train_g.num_nodes()

    entity_ids = np.arange(0, num_entities)
    event_ids = np.arange(num_entities, num_entities + num_events)

    # 构建无向图用于随机游走
    bg_train_g_undir = dgl.graph((
        torch.cat([bg_train_g.edges()[0], bg_train_g.edges()[1]]),
        torch.cat([bg_train_g.edges()[1], bg_train_g.edges()[0]])
    ))

    while True:
        # ------ 随机游走采样节点 ------
        sel_nodes = []
        for i in range(args.rw_0):
            if i == 0:
                num_event_seed = max(1, int(args.rw_1 * 0.02))
                num_entity_seed = args.rw_1 - num_event_seed
                seeds = np.concatenate([
                    np.random.choice(event_ids, num_event_seed, replace=False),
                    np.random.choice(entity_ids, num_entity_seed, replace=False)
                ])
            else:
                seeds = np.random.choice(sel_nodes, args.rw_1, replace=True)

            rw, _ = dgl.sampling.random_walk(bg_train_g_undir, seeds, length=args.rw_2)
            rw = rw.reshape(-1).cpu().numpy()
            sel_nodes.extend(np.unique(rw[rw != -1]))

        sel_nodes = list(np.unique(sel_nodes))
        if len(sel_nodes) < 50:
            continue

        # ------ 构建初始子图 ------
        sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)
        sampled_global_ids = sub_g.ndata[dgl.NID].tolist()

        event_nodes = [nid for nid in sampled_global_ids if nid in event_ids]
        entity_nodes = [nid for nid in sampled_global_ids if nid in entity_ids]

        # 必须至少有一个事件
        if len(event_nodes) == 0:
            continue

        # 限制事件比例
        max_event_ratio = 0.2
        if len(event_nodes) / len(sampled_global_ids) > max_event_ratio:
            max_events = max(1, int(len(entity_nodes) * max_event_ratio))
            keep_events = event_nodes[:max_events]
            keep_nodes = entity_nodes + keep_events
            sub_g = dgl.node_subgraph(bg_train_g, keep_nodes)
            sampled_global_ids = sub_g.ndata[dgl.NID].tolist()

        if sub_g.num_nodes() < 50:
            continue

        # ------ 构造三元组 ------
        sub_tri = torch.stack([sub_g.edges()[0], sub_g.edata['rel'], sub_g.edges()[1]]).T.tolist()
        random.shuffle(sub_tri)

        # 使用旧逻辑构建映射（仅用到的节点）
        ent_reidx = {}
        entidx = 0
        triples_reidx = []
        ent_freq = ddict(int)
        rel_freq = ddict(int)

        for h_local, r, t_local in sub_tri:
            h_global = sub_g.ndata[dgl.NID][h_local].item()
            t_global = sub_g.ndata[dgl.NID][t_local].item()

            if h_global not in ent_reidx:
                ent_reidx[h_global] = entidx
                entidx += 1
            if t_global not in ent_reidx:
                ent_reidx[t_global] = entidx
                entidx += 1

            h_new = ent_reidx[h_global]
            t_new = ent_reidx[t_global]
            triples_reidx.append([h_new, r, t_new])
            ent_freq[h_new] += 1
            ent_freq[t_new] += 1
            rel_freq[r] += 1

        inv_ent_reidx = {v: k for k, v in ent_reidx.items()}

        # ------ 查询 / 支持集划分 ------
        que_tris, sup_tris = [], []
        for idx, tri in enumerate(triples_reidx):
            h, r, t = tri
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                que_tris.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                sup_tris.append(tri)

            if len(que_tris) >= int(len(triples_reidx) * 0.1):
                break

        sup_tris.extend(triples_reidx[idx + 1:])
        if len(que_tris) < int(len(triples_reidx) * 0.05):
            continue

        #构建支持 → 查询索引
        hr2t, rt2h = get_hr2t_rt2h_sup_que(sup_tris, que_tris)

        # 返回结果
        return sup_tris, que_tris, hr2t, rt2h, inv_ent_reidx






def get_average_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0
    for i in range(sample_size):
        datum = sample_one_subgraph(args, bg_train_g)
        total_size += len(serialize(datum))
    return total_size / sample_size