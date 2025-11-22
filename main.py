import argparse
from utils import init_dir, set_seed, get_num_rel_entity
from meta_trainer import MetaTrainer
from post_trainer import PostTrainer
import os
import pickle
import torch
from subgraph import gen_subgraph_datasets
from pre_process import data2pkl
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def get_rel_nums_by_edges(rel2id, ee_edges, ev_edges, vv_edges):
    def extract_rel_ids(edge_list):
        return set(rel2id[r] for (_, _, r) in edge_list)

    rel_ids_ee = extract_rel_ids(ee_edges)
    rel_ids_ev = extract_rel_ids(ev_edges)
    rel_ids_vv = extract_rel_ids(vv_edges)

    return [len(rel_ids_ee), len(rel_ids_ev), len(rel_ids_vv)]
# torch.cuda.set_device("cpu")  # 强制切换到 CPU
# CUDA_LAUNCH_BLOCKING=1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='eventkg_v4')

    parser.add_argument('--name', default='eventkg_v4_transe', type=str)

    parser.add_argument('--step', default='meta_train', type=str, choices=['meta_train', 'fine_tune'])
    parser.add_argument('--metatrain_state', default='./state/eventkg_v4_transe/eventkg_v4_transe.best', type=str)

    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    # params for subgraph
    parser.add_argument('--num_train_subgraph', default=10000)#10000个子图
    parser.add_argument('--num_valid_subgraph', default=200)#200个子图
    parser.add_argument('--num_sample_for_estimate_size', default=50)
    parser.add_argument('--rw_0', default=16, type=int)
    parser.add_argument('--rw_1', default=24, type=int)
    parser.add_argument('--rw_2', default=6, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    # params for meta-train
    parser.add_argument('--metatrain_num_neg', default=16)
    parser.add_argument('--metatrain_num_epoch', default=2)
    parser.add_argument('--metatrain_bs', default=8, type=int)#16
    parser.add_argument('--metatrain_lr', default=0.01, type=float)
    parser.add_argument('--metatrain_check_per_step', default=10, type=int)
    parser.add_argument('--indtest_eval_bs', default=512, type=int)

    # params for fine-tune
    parser.add_argument('--posttrain_num_neg', default=64, type=int)
    parser.add_argument('--posttrain_bs', default=512, type=int)
    parser.add_argument('--posttrain_lr', default=0.001, type=int)
    parser.add_argument('--posttrain_num_epoch', default=100, type=int)
    parser.add_argument('--posttrain_check_per_epoch', default=10, type=int)

    # params for R-GCN
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_bases', default=4, type=int)
    parser.add_argument('--emb_dim', default=32, type=int)

    # params for KGE
    parser.add_argument('--kge', default='RotatE', type=str, choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'])
    parser.add_argument('--gamma', default=15, type=float)#10
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--num_ent',type=int,default=0)
    parser.add_argument('--num_rel', type=int, default=0)
    parser.add_argument('--B', type=int, default=10)
    parser.add_argument('--gpu', default='cuda:0', type=str)#cuda:0,cpu
    parser.add_argument('--seed', default=1234, type=int)

    args = parser.parse_args()
    init_dir(args)

    args.data_path = f'./data/{args.data_name}.pkl'

    if not os.path.exists(args.data_path):
        data2pkl(args.data_name)


    args.db_path = f'./data/{args.data_name}_subgraph'

    data = pickle.load(open(args.data_path, 'rb'))
    args.entity2id = data['train_dict']['entity2id']
    args.fuzzy_dict = data['train_dict']['fuzzy_dict']
    args.test_entity2id = data['test_dict']['test_entity2id']
    args.test_event2id = data['test_dict']['test_event2id']
    args.relation2id = data['train_dict']['relation2id']
    args.event2id = data['train_dict']['event2id']
    # args.num_nodes = len(args.entity2id) + len(args.event2id)
    args.ee_edges = data['train_mutiview']['ee_edges']
    args.ev_edges = data['train_mutiview']['ev_edges']
    args.vv_edges = data['train_mutiview']['vv_edges']

    args.num_nodes = len(args.entity2id) + len(args.event2id)
    args.test_ee_edges = data['test_mutiview']['test_ee_edges']
    args.test_ev_edges = data['test_mutiview']['test_ev_edges']
    args.test_vv_edges = data['test_mutiview']['test_vv_edges']
    args.test_fuzzy_dict = data['test_dict']['test_fuzzy_dict']

    # args.in_dim = 128#128

    # args.h_dim = 64
    # args.out_dim = 32
    args.rel_nums = get_rel_nums_by_edges(args.relation2id,args.ee_edges,args.ev_edges,args.vv_edges)
    args.test_rel_nums = get_rel_nums_by_edges(args.relation2id, args.test_ee_edges, args.test_ev_edges, args.test_vv_edges)



    # args.ent_dim = args.emb_dim
    # args.rel_dim = args.emb_dim
    if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
        args.rel_dim = args.emb_dim*2
    else:
        args.ent_dim = args.emb_dim
        args.rel_dim = args.emb_dim

    args.gnn_dim = args.ent_dim
    args.in_dim = args.ent_dim
    args.h_dim = args.ent_dim
    args.out_dim = args.ent_dim



    # specify the paths for original data and subgraph db
    # data2pkl(args.data_name)
    # gen_subgraph_datasets(args)
    # load original data and make index


    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)

    args.num_ent,args.num_rel = get_num_rel_entity(args)

    set_seed(args.seed)

    if args.step == 'meta_train':
        meta_trainer = MetaTrainer(args)
        meta_trainer.train()#?
    elif args.step == 'fine_tune':
        post_trainer = PostTrainer(args)
        post_trainer.train()#?


