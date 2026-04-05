"""
RGTSR v3 Quick Test - ~3-5 min to verify code correctness.
"""
import os, json, time, argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from rgtsr_utils import RGTSRData
from rgtsr_model import RGTSR
from rule_guided_neighbor import RuleGuidedNeighborFinder
from segment import segment_rank_fil


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


def quick_test(args):
    device = f'cuda:{args.device}' if (torch.cuda.is_available() and args.device >= 0) else 'cpu'
    print(f"Device: {device}")

    data = RGTSRData(os.path.join(args.data_dir, args.dataset), add_reverse=True, data_format=args.data_format)
    print(f"Data: {data.num_entities} ent, {data.num_relations} rel")

    rules = {}
    if args.rules_file:
        with open(args.rules_file) as f: rules = {int(k):v for k,v in json.load(f).items()}
    filtered = {}
    for rel, rlist in rules.items():
        f = [r for r in rlist if r['conf'] >= 0.01 and r.get('body_supp',0) >= 2]
        if f: filtered[int(rel)] = f[:10]
    print(f"Rules: {sum(len(v) for v in filtered.values())} (filtered top-10/rel)")

    tg = 1 if 'yago' in args.dataset.lower() else 24
    adj = data.get_adj_dict()
    nf = RuleGuidedNeighborFinder(adj, sampling=args.sampling, max_time=data.max_time,
                                   num_entities=data.num_entities, weight_factor=2,
                                   time_granularity=tg, rule_beta=args.rule_beta)

    model = RGTSR(nf, rules_dict=filtered, num_entity=data.num_entities, num_rel=data.num_relations,
                   emb_dim=[64,32,16,8], DP_steps=3, DP_num_edges=10, alpha=0.5,
                   max_attended_edges=20, ratio_update=0, device=device,
                   emb_static_ratio=1, use_time_embedding=True)
    model.to(device)
    model.entity_raw_embed.cpu()
    model.relation_raw_embed.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = data.train_idx[data.train_idx[:,3] >= 48][:2000]
    val_data = data.valid_idx[:500]
    sp2o = data.get_sp2o()
    val_spt2o = data.get_spt2o('valid')

    print(f"\nQuick test: {len(train_data)} train, {len(val_data)} val")
    print(f"Config: emb=[64,32,16,8], bs=32, sampling={args.sampling}, beta={args.rule_beta}")
    print("=" * 60)

    losses = []
    for epoch in range(2):
        model.train()
        loader = DataLoader(train_data, batch_size=32, collate_fn=collate_wrapper, shuffle=True)
        rloss = 0
        t0 = time.time()

        for bi, sample in enumerate(loader):
            optimizer.zero_grad()
            scores, entities = model(sample)
            loss = model.loss(scores, entities, sample.target_idx, 32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rloss += loss.item()
            if bi % 10 == 0:
                print(f"  E{epoch} b{bi}/{len(loader)}: loss={loss.item():.4f}")

        avg_loss = rloss / (bi + 1)
        losses.append(avg_loss)

        # Print learned rule_gamma
        gammas = [af.rule_gamma.item() for af in model.att_flow_list]
        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, time={time.time()-t0:.1f}s, gammas={[f'{g:.3f}' for g in gammas]}")

        # Validation
        model.eval()
        vloader = DataLoader(val_data, batch_size=32, collate_fn=collate_wrapper, shuffle=False)
        h1 = h10 = mrr = nq = found = 0
        with torch.no_grad():
            for sample in vloader:
                nq += len(sample.src_idx)
                scores, entities = model(sample)
                rl, fm, _, rflt = segment_rank_fil(scores, entities, sample.target_idx,
                                                    sp2o, val_spt2o, sample.src_idx, sample.rel_idx, sample.ts)
                h1 += np.sum(rflt <= 1); h10 += np.sum(rflt <= 10)
                mrr += np.sum(1/rflt); found += np.sum(fm)

        print(f"  Val: H@1={h1/nq:.4f}, H@10={h10/nq:.4f}, MRR={mrr/nq:.4f}, Found={found/nq:.4f}")

    # Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS:")
    found_rate = found / nq
    final_mrr = mrr / nq
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    gamma_changed = any(abs(af.rule_gamma.item() - 1.0) > 0.001 for af in model.att_flow_list)

    checks = [
        ("Loss < 0.5", losses[-1] < 0.5, f"loss={losses[-1]:.4f}"),
        ("Found rate > 30%", found_rate > 0.3, f"found={found_rate:.4f}"),
        ("MRR > 0.05", final_mrr > 0.05, f"MRR={final_mrr:.4f}"),
        ("No NaN", not np.isnan(losses[-1]), f"loss={losses[-1]}"),
        ("Gradients flowing", has_grad, ""),
        ("Rule gamma learned", gamma_changed, f"gammas={[f'{af.rule_gamma.item():.3f}' for af in model.att_flow_list]}"),
    ]

    ok = True
    for name, passed, detail in checks:
        s = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {s}: {name} {detail}")
        if not passed: ok = False

    if ok: print("\n✅ All checks passed! Safe to full train.")
    else: print("\n❌ Some checks failed.")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ICEWS14')
    p.add_argument('--data_dir', default='../data/')
    p.add_argument('--data_format', default='tlogic')
    p.add_argument('--rules_file', default=None)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--sampling', type=int, default=5)
    p.add_argument('--rule_beta', type=float, default=0.5)
    args = p.parse_args()
    torch.manual_seed(12); np.random.seed(12)
    print("🧪 RGTSR v3 Quick Sanity Test\n")
    quick_test(args)
