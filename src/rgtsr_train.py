"""
RGTSR Training Pipeline v3.
Matches v3 model with rule-aware attention.
Uses ReduceLROnPlateau instead of cosine (more stable).
"""

import os, json, time, argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from rgtsr_utils import RGTSRData, store_edges
from rgtsr_model import RGTSR
from rule_guided_neighbor import RuleGuidedNeighborFinder
from segment import segment_rank_fil
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics


class SimpleCustomBatch:
    def __init__(self, data):
        t = list(zip(*data))
        self.src_idx = np.array(t[0], dtype=np.int32)
        self.rel_idx = np.array(t[1], dtype=np.int32)
        self.target_idx = np.array(t[2], dtype=np.int32)
        self.ts = np.array(t[3], dtype=np.int32)
        self.event_idx = np.array(t[-1], dtype=np.int32)
    def pin_memory(self): return self

def collate_wrapper(batch): return SimpleCustomBatch(batch)

def prepare_inputs(data, dataset='train', start_time=0):
    if dataset == 'train': events = data.train_idx
    elif dataset == 'valid': events = data.valid_idx
    elif dataset == 'test': events = data.test_idx
    else: raise ValueError
    return events[events[:, 3] >= start_time]


def run_stage1(args, data):
    print("=" * 60)
    print("STAGE 1: Rule Learning")
    print("=" * 60)
    tw = Temporal_Walk(data.train_idx, data.inv_relation_id, args.transition_distr)
    rl = Rule_Learner(tw.edges, data.id2relation, data.inv_relation_id, args.dataset)
    all_rels = sorted(tw.edges)
    if args.seed: np.random.seed(args.seed)
    t0 = time.time()
    for k, rel in enumerate(all_rels):
        for length in args.rule_lengths:
            for _ in range(args.num_walks):
                ok, walk = tw.sample_walk(length + 1, rel)
                if ok: rl.create_rule(walk)
            if (k+1) % 50 == 0:
                print(f"  {k+1}/{len(all_rels)} relations done")
    print(f"Rule learning: {time.time()-t0:.0f}s")
    rl.sort_rules_dict()
    rules_statistics(rl.rules_dict)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{datetime.now().strftime('%d%m%y%H%M%S')}_rules.json")
    with open(path, "w") as f: json.dump({int(k):v for k,v in rl.rules_dict.items()}, f)
    print(f"Saved: {path}")
    return rl.rules_dict, path


def run_stage2(args, data, rules_dict):
    print("=" * 60)
    print("STAGE 2: Rule-Aware Neural Training (v3)")
    print("=" * 60)

    device = f'cuda:{args.device}' if (torch.cuda.is_available() and args.device >= 0) else 'cpu'
    print(f"Device: {device}")

    adj = data.get_adj_dict()
    tg = 1 if 'yago' in args.dataset.lower() else 24

    nf = RuleGuidedNeighborFinder(adj, sampling=args.sampling, max_time=data.max_time,
                                   num_entities=data.num_entities, weight_factor=args.weight_factor,
                                   time_granularity=tg, rule_beta=args.rule_beta)

    # Filter rules
    filtered = {}
    for rel, rlist in rules_dict.items():
        rel = int(rel)
        f = [r for r in rlist if r['conf'] >= args.min_rule_conf and r.get('body_supp', 0) >= args.min_body_supp]
        if f: filtered[rel] = sorted(f, key=lambda x: x['conf'], reverse=True)
    print(f"Rules: {sum(len(v) for v in filtered.values())} for {len(filtered)} relations")

    model = RGTSR(nf, rules_dict=filtered, num_entity=data.num_entities, num_rel=data.num_relations,
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
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)

    ckpt_dir = os.path.join(args.output_dir, args.dataset, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    sp2o = data.get_sp2o()
    val_spt2o = data.get_spt2o('valid')
    train_inputs = prepare_inputs(data, 'train', start_time=args.warm_start_time)
    val_inputs = prepare_inputs(data, 'valid')

    # Print rule_gamma initial value
    for i, af in enumerate(model.att_flow_list):
        print(f"  Step {i} rule_gamma init: {af.rule_gamma.item():.4f}")

    best_epoch, best_mrr = 0, 0

    for epoch in range(args.epoch):
        print(f"\n--- Epoch {epoch} ---")
        t0 = time.time()
        loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, shuffle=True)
        model.train()
        running_loss = 0.0

        for bi, sample in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            scores, entities = model(sample)
            loss = model.loss(scores, entities, sample.target_idx, args.batch_size,
                              args.gradient_iters_per_update, args.loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (bi + 1) % args.gradient_iters_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / (bi + 1)
        lr_now = optimizer.param_groups[0]['lr']

        # Print learned rule_gamma values
        gamma_str = ", ".join([f"{af.rule_gamma.item():.3f}" for af in model.att_flow_list])
        print(f"  Loss: {avg_loss:.6f}, LR: {lr_now:.6f}, Gammas: [{gamma_str}]")
        print(f"  Time: {time.time()-t0:.0f}s")

        # Save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': avg_loss, 'args': args, 'rules_dict': filtered
                     }, os.path.join(ckpt_dir, f'checkpoint_{epoch}.pt'))

        # Validation
        val_mrr = validate(model, val_inputs, args, sp2o, val_spt2o, device)
        scheduler.step(val_mrr)

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_epoch = epoch
            print(f"  ** New best MRR: {best_mrr:.6f} at epoch {best_epoch}")

    print(f"\nDone. Best: epoch {best_epoch}, MRR {best_mrr:.6f}")
    return model, best_epoch


def validate(model, val_inputs, args, sp2o, val_spt2o, device):
    model.eval()
    loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, shuffle=False)
    h1 = h3 = h10 = mrr = nq = found = 0

    with torch.no_grad():
        for sample in loader:
            nq += len(sample.src_idx)
            scores, entities = model(sample)
            rl, fm, rfl, rflt = segment_rank_fil(scores, entities, sample.target_idx,
                                                  sp2o, val_spt2o, sample.src_idx, sample.rel_idx, sample.ts)
            h1 += np.sum(rflt <= 1)
            h3 += np.sum(rflt <= 3)
            h10 += np.sum(rflt <= 10)
            mrr += np.sum(1/rflt)
            found += np.sum(fm)

    print(f"  Val (filtered): H@1={h1/nq:.4f}, H@3={h3/nq:.4f}, H@10={h10/nq:.4f}, MRR={mrr/nq:.4f}")
    print(f"  Val (raw): Found={found/nq:.4f}")
    return mrr / nq


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='../data/')
    p.add_argument('--data_format', type=str, default='tlogic')
    p.add_argument('--output_dir', type=str, default='../output/')
    p.add_argument('--stage', type=str, default='both', choices=['1','2','both'])
    p.add_argument('--rules_file', type=str, default=None)
    p.add_argument('--rule_lengths', type=int, nargs='+', default=[1,2,3])
    p.add_argument('--num_walks', type=int, default=100)
    p.add_argument('--transition_distr', type=str, default='exp')
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--emb_dim', type=int, nargs='+', default=[256,128,64,32])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--DP_steps', type=int, default=3)
    p.add_argument('--DP_num_edges', type=int, default=15)
    p.add_argument('--max_attended_edges', type=int, default=40)
    p.add_argument('--ratio_update', type=float, default=0)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epoch', type=int, default=20)
    p.add_argument('--device', type=int, default=-1)
    p.add_argument('--sampling', type=int, default=5)
    p.add_argument('--weight_factor', type=float, default=2)
    p.add_argument('--node_score_aggregation', type=str, default='sum')
    p.add_argument('--ent_score_aggregation', type=str, default='sum')
    p.add_argument('--emb_static_ratio', type=float, default=1)
    p.add_argument('--loss_fn', type=str, default='BCE')
    p.add_argument('--no_time_embedding', action='store_true', default=False)
    p.add_argument('--diac_embed', action='store_true')
    p.add_argument('--warm_start_time', type=int, default=48)
    p.add_argument('--gradient_iters_per_update', type=int, default=1)
    p.add_argument('--val_every', type=int, default=1)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--rule_beta', type=float, default=0.5)
    p.add_argument('--min_rule_conf', type=float, default=0.01)
    p.add_argument('--min_body_supp', type=int, default=2)

    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print(f"RGTSR v3 — Rule-Aware Attention")
    print(f"Dataset: {args.dataset}, Stage: {args.stage}")
    print(f"alpha={args.alpha}, beta={args.rule_beta}, sampling={args.sampling}\n")

    data = RGTSRData(os.path.join(args.data_dir, args.dataset), add_reverse=True, data_format=args.data_format)
    print(f"Data: {data.num_entities} ent, {data.num_relations} rel, {len(data.train_idx)} train\n")

    rules = {}
    if args.stage in ['1','both']:
        rules, _ = run_stage1(args, data)
    if args.stage in ['2','both']:
        if args.stage == '2' and args.rules_file:
            with open(args.rules_file) as f: rules = {int(k):v for k,v in json.load(f).items()}
            print(f"Loaded {sum(len(v) for v in rules.values())} rules")
        run_stage2(args, data, rules)
    print("\nAll done!")
