"""
RGTSR Evaluation Module.

Supports:
  - Neural-only evaluation (xERTE-style attention scores)
  - Rule-only evaluation (TLogic-style rule application)
  - Combined evaluation (α * neural + (1-α) * rule)
  - Inductive fallback: unseen entities use pure rule-based scoring
  - Filtered metrics (time-dependent and time-independent)

Usage:
  python rgtsr_eval.py --dataset ICEWS14 --checkpoint checkpoints/checkpoint_best.pt
  python rgtsr_eval.py --dataset ICEWS14 --checkpoint ... --eval_mode combined --alpha 0.5
"""

import os
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from rgtsr_utils import RGTSRData, store_edges, calculate_obj_distribution
from rgtsr_model import RGTSR
from rule_guided_neighbor import RuleGuidedNeighborFinder
from segment import segment_rank_fil, segment_norm_l1

# TLogic rule application (reused)
import rule_application as ra
from score_functions import score_12


class SimpleCustomBatch:
    def __init__(self, data):
        transposed = list(zip(*data))
        self.src_idx = np.array(transposed[0], dtype=np.int32)
        self.rel_idx = np.array(transposed[1], dtype=np.int32)
        self.target_idx = np.array(transposed[2], dtype=np.int32)
        self.ts = np.array(transposed[3], dtype=np.int32)
        self.event_idx = np.array(transposed[-1], dtype=np.int32)


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def apply_rules_for_query(query, rules_dict, edges, score_func, score_args):
    """
    Apply TLogic-style rule application for a single query.

    Parameters:
        query: np.array [sub, rel, obj, ts]
        rules_dict: dict, relation -> list of rules
        edges: dict, relation -> edges array
        score_func: scoring function
        score_args: arguments for scoring function

    Returns:
        candidates: dict, entity_id -> confidence score
    """
    test_query_sub = query[0]
    test_query_rel = query[1]
    test_query_ts = query[3]

    if test_query_rel not in rules_dict:
        return {}

    cands_dict = {}
    for rule in rules_dict[test_query_rel]:
        walk_edges = ra.match_body_relations(rule, edges, test_query_sub)

        if 0 not in [len(x) for x in walk_edges]:
            rule_walks = ra.get_walks(rule, walk_edges)
            if rule['var_constraints']:
                rule_walks = ra.check_var_constraints(rule['var_constraints'], rule_walks)

            if not rule_walks.empty:
                max_entity = "entity_" + str(len(rule['body_rels']))
                cands = set(rule_walks[max_entity])
                for cand in cands:
                    cands_walks = rule_walks[rule_walks[max_entity] == cand]
                    score = score_func(
                        rule, cands_walks, test_query_ts,
                        *score_args
                    ).astype(np.float32)
                    try:
                        cands_dict[cand].append(score)
                    except KeyError:
                        cands_dict[cand] = [score]

    # Apply noisy-or aggregation
    if cands_dict:
        scores = {
            cand: float(1 - np.prod(1 - np.array(scores_list)))
            for cand, scores_list in cands_dict.items()
        }
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return {}


def evaluate(model, data, test_inputs, args, sp2o, test_spt2o, device,
             rules_dict=None, learn_edges=None):
    """
    Full evaluation with optional combined scoring.

    Parameters:
        model: RGTSR model (can be None for rule-only eval)
        data: RGTSRData instance
        test_inputs: np.array, test quadruples
        args: arguments
        sp2o: dict, (s,p) -> [o]
        test_spt2o: dict, (s,p,t) -> [o]
        device: str
        rules_dict: dict, for rule-based scoring
        learn_edges: dict, edges for rule application
    """
    eval_mode = args.eval_mode
    alpha = args.alpha

    print(f"\nEvaluation mode: {eval_mode}, alpha: {alpha}")
    print(f"Test samples: {len(test_inputs)}")

    # Metrics
    hit_1_raw = hit_3_raw = hit_10_raw = 0
    hit_1_fil = hit_3_fil = hit_10_fil = 0
    hit_1_fil_t = hit_3_fil_t = hit_10_fil_t = 0
    mrr_raw = mrr_fil = mrr_fil_t = 0
    found_cnt = 0
    num_query = 0
    rule_fallback_cnt = 0

    if eval_mode in ['neural', 'combined']:
        model.eval()
        test_loader = DataLoader(
            test_inputs, batch_size=args.batch_size,
            collate_fn=collate_wrapper, shuffle=False
        )

        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(test_loader), desc="Evaluating"):
                src_idx_l = sample.src_idx
                rel_idx_l = sample.rel_idx
                target_idx_l = sample.target_idx
                cut_time_l = sample.ts
                batch_size = len(src_idx_l)
                num_query += batch_size

                # Neural forward pass
                entity_att_score, entities = model(sample)

                # Combined scoring
                if eval_mode == 'combined' and rules_dict:
                    entity_att_score = model.combined_score(
                        entity_att_score, entities, rel_idx_l, alpha=alpha
                    )

                # Compute ranks
                target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = \
                    segment_rank_fil(
                        entity_att_score, entities, target_idx_l,
                        sp2o, test_spt2o, src_idx_l, rel_idx_l, cut_time_l
                    )

                hit_1_raw += np.sum(target_rank_l == 1)
                hit_3_raw += np.sum(target_rank_l <= 3)
                hit_10_raw += np.sum(target_rank_l <= 10)
                hit_1_fil += np.sum(target_rank_fil_l <= 1)
                hit_3_fil += np.sum(target_rank_fil_l <= 3)
                hit_10_fil += np.sum(target_rank_fil_l <= 10)
                hit_1_fil_t += np.sum(target_rank_fil_t_l <= 1)
                hit_3_fil_t += np.sum(target_rank_fil_t_l <= 3)
                hit_10_fil_t += np.sum(target_rank_fil_t_l <= 10)
                found_cnt += np.sum(found_mask)
                mrr_raw += np.sum(1 / target_rank_l)
                mrr_fil += np.sum(1 / target_rank_fil_l)
                mrr_fil_t += np.sum(1 / target_rank_fil_t_l)

    elif eval_mode == 'rule_only':
        # Pure rule-based evaluation (TLogic style)
        assert rules_dict is not None, "rules_dict required for rule_only mode"
        assert learn_edges is not None, "learn_edges required for rule_only mode"

        score_func = score_12
        score_args = [0.1, 0.5]

        obj_dist, rel_obj_dist = calculate_obj_distribution(data.train_idx, learn_edges)

        for i in tqdm(range(len(test_inputs)), desc="Rule evaluation"):
            query = test_inputs[i]
            num_query += 1

            # Apply rules
            candidates = apply_rules_for_query(
                query, rules_dict, learn_edges, score_func, score_args
            )

            if not candidates:
                # Fallback to object distribution baseline
                rule_fallback_cnt += 1
                test_query_rel = query[1]
                if test_query_rel in learn_edges:
                    candidates = rel_obj_dist.get(test_query_rel, obj_dist)
                else:
                    candidates = obj_dist

            # Filter other valid answers
            candidates = _filter_candidates(query, candidates, test_inputs)

            # Calculate rank
            rank = _calculate_rank(query[2], candidates, data.num_entities)

            if rank <= 10:
                hit_10_raw += 1
                hit_10_fil += 1
                hit_10_fil_t += 1
                if rank <= 3:
                    hit_3_raw += 1
                    hit_3_fil += 1
                    hit_3_fil_t += 1
                    if rank == 1:
                        hit_1_raw += 1
                        hit_1_fil += 1
                        hit_1_fil_t += 1
            mrr_raw += 1 / rank
            mrr_fil += 1 / rank
            mrr_fil_t += 1 / rank
            found_cnt += 1

        print(f"  Rule fallback (baseline): {rule_fallback_cnt}/{num_query}")

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Filtered (time-dep):  H@1={hit_1_fil_t/num_query:.6f}  "
          f"H@3={hit_3_fil_t/num_query:.6f}  "
          f"H@10={hit_10_fil_t/num_query:.6f}  "
          f"MRR={mrr_fil_t/num_query:.6f}")
    print(f"Filtered (time-ind):  H@1={hit_1_fil/num_query:.6f}  "
          f"H@3={hit_3_fil/num_query:.6f}  "
          f"H@10={hit_10_fil/num_query:.6f}  "
          f"MRR={mrr_fil/num_query:.6f}")
    print(f"Raw:                  H@1={hit_1_raw/num_query:.6f}  "
          f"H@3={hit_3_raw/num_query:.6f}  "
          f"H@10={hit_10_raw/num_query:.6f}  "
          f"MRR={mrr_raw/num_query:.6f}  "
          f"Found={found_cnt/num_query:.6f}")

    # Save results
    results = {
        'eval_mode': eval_mode,
        'alpha': alpha,
        'num_queries': num_query,
        'filtered_time_dep': {
            'hits@1': hit_1_fil_t / num_query,
            'hits@3': hit_3_fil_t / num_query,
            'hits@10': hit_10_fil_t / num_query,
            'mrr': mrr_fil_t / num_query
        },
        'filtered_time_ind': {
            'hits@1': hit_1_fil / num_query,
            'hits@3': hit_3_fil / num_query,
            'hits@10': hit_10_fil / num_query,
            'mrr': mrr_fil / num_query
        },
        'raw': {
            'hits@1': hit_1_raw / num_query,
            'hits@3': hit_3_raw / num_query,
            'hits@10': hit_10_raw / num_query,
            'mrr': mrr_raw / num_query,
            'found_rate': found_cnt / num_query
        }
    }

    return results


def _filter_candidates(query, candidates, test_data):
    """Filter out other valid answers for the same query (for fair evaluation)."""
    other_answers = test_data[
        (test_data[:, 0] == query[0]) &
        (test_data[:, 1] == query[1]) &
        (test_data[:, 2] != query[2]) &
        (test_data[:, 3] == query[3])
    ]
    if len(other_answers):
        for obj in other_answers[:, 2]:
            candidates.pop(obj, None)
    return candidates


def _calculate_rank(answer, candidates, num_entities, setting='best'):
    """Calculate rank of correct answer among candidates."""
    rank = num_entities
    if answer in candidates:
        conf = candidates[answer]
        all_confs = list(candidates.values())
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]
        if setting == 'best':
            rank = ranks[0] + 1
        elif setting == 'worst':
            rank = ranks[-1] + 1
        elif setting == 'average':
            rank = (ranks[0] + ranks[-1]) // 2 + 1
    return rank


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGTSR Evaluation")

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--data_format', type=str, default='xerte', choices=['tlogic', 'xerte'])
    parser.add_argument('--output_dir', type=str, default='../output/')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--rules_file', type=str, default=None,
                        help='Path to rules JSON file')

    parser.add_argument('--eval_mode', type=str, default='combined',
                        choices=['neural', 'rule_only', 'combined'],
                        help='Evaluation mode')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for combined scoring')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--whole_or_seen', type=str, default='whole',
                        choices=['whole', 'seen', 'unseen'])

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available() and args.device >= 0:
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'

    # Load data
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    data = RGTSRData(dataset_dir, add_reverse=True, data_format=args.data_format)

    # Select test set
    if args.whole_or_seen == 'whole':
        test_data = data.test_idx
    elif args.whole_or_seen == 'seen':
        test_data = data.test_idx_seen
    elif args.whole_or_seen == 'unseen':
        test_data = data.test_idx_unseen

    sp2o = data.get_sp2o()
    test_spt2o = data.get_spt2o('test')

    # Load rules
    rules_dict = None
    learn_edges = None
    if args.rules_file:
        with open(args.rules_file) as f:
            rules_dict = json.load(f)
        rules_dict = {int(k): v for k, v in rules_dict.items()}
        learn_edges = store_edges(data.train_idx)
        print(f"Loaded {sum(len(v) for v in rules_dict.values())} rules")

    # Load model
    model = None
    if args.eval_mode in ['neural', 'combined'] and args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_args = checkpoint['args']

        # Determine time granularity
        if 'yago' in args.dataset.lower():
            time_granularity = 1
        elif 'icews' in args.dataset.lower():
            time_granularity = 24
        else:
            time_granularity = 24

        adj = data.get_adj_dict()
        nf = RuleGuidedNeighborFinder(
            adj, sampling=getattr(ckpt_args, 'sampling', 5),
            max_time=data.max_time, num_entities=data.num_entities,
            weight_factor=getattr(ckpt_args, 'weight_factor', 2),
            time_granularity=time_granularity,
            rule_beta=getattr(ckpt_args, 'rule_beta', 2.0)
        )

        # Load rules from checkpoint if not provided separately
        if rules_dict is None and 'rules_dict' in checkpoint:
            rules_dict = checkpoint['rules_dict']

        model = RGTSR(
            nf, rules_dict=rules_dict,
            num_entity=data.num_entities,
            num_rel=data.num_relations,
            emb_dim=ckpt_args.emb_dim,
            DP_steps=ckpt_args.DP_steps,
            DP_num_edges=ckpt_args.DP_num_edges,
            alpha=args.alpha,
            max_attended_edges=ckpt_args.max_attended_edges,
            node_score_aggregation=ckpt_args.node_score_aggregation,
            ent_score_aggregation=ckpt_args.ent_score_aggregation,
            ratio_update=ckpt_args.ratio_update,
            device=device,
            diac_embed=ckpt_args.diac_embed,
            emb_static_ratio=ckpt_args.emb_static_ratio,
            use_time_embedding=not ckpt_args.no_time_embedding
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.entity_raw_embed.cpu()
        model.relation_raw_embed.cpu()
        print(f"Loaded model from {args.checkpoint} (epoch {checkpoint['epoch']})")

    # Run evaluation
    results = evaluate(
        model, data, test_data, args, sp2o, test_spt2o, device,
        rules_dict=rules_dict, learn_edges=learn_edges
    )

    # Save results
    results_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'eval_{args.eval_mode}_{args.whole_or_seen}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
