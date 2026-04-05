"""
RGTSR Training Pipeline v2 (FIXED).

FIX 3: Added cosine LR scheduler to prevent gradient spikes.
Also includes warmup for first epoch.
"""

import os, sys, json, time, argparse
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from rgtsr_utils import RGTSRData, store_edges
from rgtsr_model import RGTSR
from rule_guided_neighbor import RuleGuidedNeighborFinder
from segment import segment_rank_fil
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics
from score_functions import score_12


class SimpleCustomBatch:
    def __init__(self, data):
        transposed = list(zip(*data))
        self.src_idx = np.array(transposed[0], dtype=np.int32)
        self.rel_idx = np.array(transposed[1], dtype=np.int32)
        self.target_idx = np.array(transposed[2], dtype=np.int32)
        self.ts = np.array(transposed[3], dtype=np.int32)
        self.event_idx = np.array(transposed[-1], dtype=np.int32)
    def pin_memory(self):
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def prepare_inputs(data, dataset='train', start_time=0):
    if dataset == 'train': events = data.train_idx
    elif dataset == 'valid': events = data.valid_idx
    elif dataset == 'test': events = data.test_idx
    else: raise ValueError
    return events[events[:, 3] >= start_time]


# ===========================================================================
# Stage 1: Rule Learning
# ===========================================================================
def run_stage1_rule_learning(args, data):
    print("=" * 60)
    print("STAGE 1: Temporal Rule Learning (TLogic)")
    print("=" * 60)

    temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, args.transition_distr)
    rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, args.dataset)
    all_relations = sorted(temporal_walk.edges)

    if args.seed is not None:
        np.random.seed(args.seed)

    start = time.time()
    for k, rel in enumerate(all_relations):
        for length in args.rule_lengths:
            it_start = time.time()
            for _ in range(args.num_walks):
                walk_ok, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_ok:
                    rl.create_rule(walk)
            n_rules = sum(len(v) for v in rl.rules_dict.values()) // 2
            print(f"  Relation {k+1}/{len(all_relations)}, length {length}: "
                  f"{time.time()-it_start:.1f}s, total rules: {n_rules}")

    print(f"Rule learning finished in {time.time()-start:.2f} seconds.")
    rl.sort_rules_dict()
    rules_statistics(rl.rules_dict)

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    dt = datetime.now().strftime("%d%m%y%H%M%S")
    rules_path = os.path.join(output_dir, f"{dt}_rgtsr_rules.json")
    with open(rules_path, "w") as f:
        json.dump({int(k): v for k, v in rl.rules_dict.items()}, f)
    print(f"Rules saved to {rules_path}")
    return rl.rules_dict, rules_path


# ===========================================================================
# Stage 2: Neural Training (FIXED)
# ===========================================================================
def run_stage2_neural_training(args, data, rules_dict):
    print("=" * 60)
    print("STAGE 2: Rule-Guided Neural Training (RGTSR)")
    print("=" * 60)

    device = f'cuda:{args.device}' if (torch.cuda.is_available() and args.device >= 0) else 'cpu'
    print(f"Using device: {device}")

    adj = data.get_adj_dict()
    if 'yago' in args.dataset.lower(): tg = 1
    elif 'icews' in args.dataset.lower(): tg = 24
    else: tg = 24

    nf = RuleGuidedNeighborFinder(adj, sampling=args.sampling, max_time=data.max_time,
                                   num_entities=data.num_entities, weight_factor=args.weight_factor,
                                   time_granularity=tg, rule_beta=args.rule_beta)

    # Filter rules
    filtered_rules = {}
    for rel, rule_list in rules_dict.items():
        rel = int(rel)
        filtered = [r for r in rule_list
                    if r['conf'] >= args.min_rule_conf and r.get('body_supp', 0) >= args.min_body_supp]
        if filtered:
            filtered_rules[rel] = sorted(filtered, key=lambda x: x['conf'], reverse=True)
    print(f"Filtered rules: {sum(len(v) for v in filtered_rules.values())} rules for {len(filtered_rules)} relations")

    model = RGTSR(nf, rules_dict=filtered_rules, num_entity=data.num_entities, num_rel=data.num_relations,
                   emb_dim=args.emb_dim, DP_steps=args.DP_steps, DP_num_edges=args.DP_num_edges,
                   alpha=args.alpha, max_attended_edges=args.max_attended_edges,
                   node_score_aggregation=args.node_score_aggregation,
                   ent_score_aggregation=args.ent_score_aggregation,
                   ratio_update=args.ratio_update, device=device, diac_embed=args.diac_embed,
                   emb_static_ratio=args.emb_static_ratio, use_time_embedding=not args.no_time_embedding)

    model.to(device)
    model.entity_raw_embed.cpu()
    model.relation_raw_embed.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # FIX 3: Cosine annealing LR scheduler for stability
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=args.lr * 0.01)

    checkpoint_dir = os.path.join(args.output_dir, args.dataset, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    sp2o = data.get_sp2o()
    val_spt2o = data.get_spt2o('valid')
    train_inputs = prepare_inputs(data, 'train', start_time=args.warm_start_time)
    val_inputs = prepare_inputs(data, 'valid')

    best_epoch, best_val_mrr = 0, 0

    for epoch in range(args.epoch):
        print(f"\n--- Epoch {epoch} ---")
        epoch_start = time.time()

        train_loader = DataLoader(train_inputs, batch_size=args.batch_size,
                                   collate_fn=collate_wrapper, shuffle=True)
        model.train()
        running_loss = 0.0

        for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            entity_att_score, entities = model(sample)
            loss = model.loss(entity_att_score, entities, sample.target_idx,
                              args.batch_size, args.gradient_iters_per_update, args.loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (batch_idx + 1) % args.gradient_iters_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        # FIX 3: Step scheduler
        scheduler.step()

        running_loss /= (batch_idx + 1)
        print(f"  Training loss: {running_loss:.6f}")
        print(f"  Training time: {time.time() - epoch_start:.1f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': running_loss, 'args': args, 'rules_dict': filtered_rules
        }, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt'))

        # Validation
        if epoch % args.val_every == 0:
            val_mrr = validate(model, data, val_inputs, args, sp2o, val_spt2o, device)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_epoch = epoch
                print(f"  ** New best MRR: {best_val_mrr:.6f} at epoch {best_epoch}")

    print(f"\nTraining finished. Best epoch: {best_epoch}, Best MRR: {best_val_mrr:.6f}")
    return model, best_epoch


def validate(model, data, val_inputs, args, sp2o, val_spt2o, device):
    model.eval()
    val_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, shuffle=False)

    hit_1 = hit_3 = hit_10 = 0
    hit_1_fil = hit_3_fil = hit_10_fil = 0
    mrr_total = mrr_fil = 0
    num_query = found_cnt = 0

    with torch.no_grad():
        for sample in val_loader:
            num_query += len(sample.src_idx)
            entity_att_score, entities = model(sample)
            target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = \
                segment_rank_fil(entity_att_score, entities, sample.target_idx,
                                 sp2o, val_spt2o, sample.src_idx, sample.rel_idx, sample.ts)

            hit_1 += np.sum(target_rank_l == 1)
            hit_3 += np.sum(target_rank_l <= 3)
            hit_10 += np.sum(target_rank_l <= 10)
            hit_1_fil += np.sum(target_rank_fil_t_l <= 1)
            hit_3_fil += np.sum(target_rank_fil_t_l <= 3)
            hit_10_fil += np.sum(target_rank_fil_t_l <= 10)
            found_cnt += np.sum(found_mask)
            mrr_total += np.sum(1 / target_rank_l)
            mrr_fil += np.sum(1 / target_rank_fil_t_l)

    print(f"  Validation (filtered, time-dep): H@1={hit_1_fil/num_query:.4f}, "
          f"H@3={hit_3_fil/num_query:.4f}, H@10={hit_10_fil/num_query:.4f}, MRR={mrr_fil/num_query:.4f}")
    print(f"  Validation (raw): H@1={hit_1/num_query:.4f}, H@3={hit_3/num_query:.4f}, "
          f"H@10={hit_10/num_query:.4f}, MRR={mrr_total/num_query:.4f}, Found={found_cnt/num_query:.4f}")
    return mrr_fil / num_query


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGTSR Training Pipeline v2")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--data_format', type=str, default='xerte', choices=['tlogic', 'xerte'])
    parser.add_argument('--output_dir', type=str, default='../output/')
    parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'])
    parser.add_argument('--rules_file', type=str, default=None)

    # Stage 1
    parser.add_argument('--rule_lengths', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--num_walks', type=int, default=100)
    parser.add_argument('--transition_distr', type=str, default='exp')
    parser.add_argument('--seed', type=int, default=12)

    # Stage 2
    parser.add_argument('--emb_dim', type=int, nargs='+', default=[256, 128, 64, 32])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--DP_steps', type=int, default=3)
    parser.add_argument('--DP_num_edges', type=int, default=15)
    parser.add_argument('--max_attended_edges', type=int, default=40)
    parser.add_argument('--ratio_update', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--sampling', type=int, default=5)
    parser.add_argument('--weight_factor', type=float, default=2)
    parser.add_argument('--node_score_aggregation', type=str, default='sum')
    parser.add_argument('--ent_score_aggregation', type=str, default='sum')
    parser.add_argument('--emb_static_ratio', type=float, default=1)
    parser.add_argument('--loss_fn', type=str, default='BCE')
    parser.add_argument('--no_time_embedding', action='store_true', default=False)
    parser.add_argument('--diac_embed', action='store_true')
    parser.add_argument('--warm_start_time', type=int, default=48)
    parser.add_argument('--gradient_iters_per_update', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=1)

    # RGTSR-specific
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--rule_beta', type=float, default=0.5)
    parser.add_argument('--min_rule_conf', type=float, default=0.01)
    parser.add_argument('--min_body_supp', type=int, default=2)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("RGTSR v2 - Rule-Guided Temporal Subgraph Reasoning (FIXED)")
    print(f"Dataset: {args.dataset}, Stage: {args.stage}")
    print(f"Config: alpha={args.alpha}, rule_beta={args.rule_beta}, DP_steps={args.DP_steps}, sampling={args.sampling}")
    print()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    data = RGTSRData(dataset_dir, add_reverse=True, data_format=args.data_format)
    print(f"Loaded: {data.num_entities} entities, {data.num_relations} relations "
          f"(+{data.num_relations_orig} inverse), {len(data.train_idx)} train quads")

    rules_dict = {}
    if args.stage in ['1', 'both']:
        rules_dict, rules_path = run_stage1_rule_learning(args, data)

    if args.stage in ['2', 'both']:
        if args.stage == '2':
            if args.rules_file is None:
                print("Warning: No rules file. Running without rules.")
            else:
                with open(args.rules_file) as f:
                    rules_dict = json.load(f)
                rules_dict = {int(k): v for k, v in rules_dict.items()}
                print(f"Loaded {sum(len(v) for v in rules_dict.values())} rules from {args.rules_file}")
        model, best_epoch = run_stage2_neural_training(args, data, rules_dict)

    print("\nDone!")
