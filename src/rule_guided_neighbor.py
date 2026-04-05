"""
Rule-Guided Neighbor Finder v3.
Per-query rule matching + provides rule alignment signal for attention.
"""

import numpy as np


class RuleGuidedNeighborFinder:
    def __init__(self, adj, sampling=5, max_time=366*24, num_entities=None,
                 weight_factor=1, time_granularity=24, rule_beta=0.5):
        self.time_granularity = time_granularity
        self.sampling = sampling
        self.weight_factor = weight_factor
        self.rule_beta = rule_beta
        self.per_query_matching_rels = {}  # eg_idx -> set of matching rel ids at current hop

        n_idx, n_ts, e_idx, off_set, off_set_t = self._init_off_set(adj, max_time, num_entities)
        self.node_idx_l = n_idx
        self.node_ts_l = n_ts
        self.edge_idx_l = e_idx
        self.off_set_l = off_set
        self.off_set_t_l = off_set_t

    def _init_off_set(self, adj, max_time, num_entities):
        n_idx, n_ts, e_idx = [], [], []
        off_set = [0]
        off_set_t = []
        for i in range(num_entities):
            curr = sorted(adj.get(i, []), key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
            n_idx.extend([x[0] for x in curr])
            e_idx.extend([x[1] for x in curr])
            curr_ts = [x[2] for x in curr]
            n_ts.extend(curr_ts)
            off_set.append(len(n_idx))
            off_set_t.append([np.searchsorted(curr_ts, ct, 'left')
                              for ct in range(0, max_time + 1, self.time_granularity)])
        return np.array(n_idx), np.array(n_ts), np.array(e_idx), np.array(off_set), off_set_t

    def set_per_query_rules(self, batch_rules, current_hop=0):
        """Set per-query matching relations for given hop."""
        self.per_query_matching_rels = {}
        for eg_idx, rules in batch_rules.items():
            matching = set()
            for rule in rules:
                if current_hop < len(rule['body_rels']):
                    matching.add(rule['body_rels'][current_hop])
            if matching:
                self.per_query_matching_rels[eg_idx] = matching

    def get_temporal_degree(self, src_idx_l, cut_time_l):
        return np.array([self.off_set_t_l[s][int(t / self.time_granularity)]
                         for s, t in zip(src_idx_l, cut_time_l)])

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, eg_idx_l=None, num_neighbors=20):
        """Sample neighbors. Returns (nodes, relations, timestamps, rule_mask).
        rule_mask[i,j] = 1.0 if neighbor j of node i matches a rule, else 0.0.
        """
        B = len(src_idx_l)
        out_node = -np.ones((B, num_neighbors), dtype=np.int32)
        out_ts = np.zeros((B, num_neighbors), dtype=np.int32)
        out_eidx = -np.ones((B, num_neighbors), dtype=np.int32)
        out_rule_mask = np.zeros((B, num_neighbors), dtype=np.float32)  # NEW: rule alignment signal

        for i, (src, ct) in enumerate(zip(src_idx_l, cut_time_l)):
            nidx = self.node_idx_l[self.off_set_l[src]:self.off_set_l[src + 1]]
            nts = self.node_ts_l[self.off_set_l[src]:self.off_set_l[src + 1]]
            neidx = self.edge_idx_l[self.off_set_l[src]:self.off_set_l[src + 1]]
            mid = self.off_set_t_l[src][int(ct / self.time_granularity)]
            ngh_idx, ngh_eidx, ngh_ts = nidx[:mid], neidx[:mid], nts[:mid]

            if len(ngh_idx) == 0:
                continue

            # Get per-query matching rels
            matching_rels = set()
            if eg_idx_l is not None and self.per_query_matching_rels:
                matching_rels = self.per_query_matching_rels.get(int(eg_idx_l[i]), set())

            # Compute time weights (same as xERTE sampling=3)
            delta_t = (ngh_ts - ct) / (self.time_granularity * self.weight_factor)
            time_weights = np.exp(delta_t) + 1e-9

            # Rule boost for sampling (mild, just guides sampling not dominates)
            if self.sampling == 5 and matching_rels:
                rule_boost = np.array([1.0 + self.rule_beta if int(r) in matching_rels else 1.0
                                       for r in ngh_eidx])
                weights = time_weights * rule_boost
            else:
                weights = time_weights

            weights /= weights.sum()
            n_sample = min(len(ngh_idx), num_neighbors)
            sampled_idx = np.sort(np.random.choice(len(ngh_idx), n_sample, replace=False, p=weights))

            out_node[i, num_neighbors - n_sample:] = ngh_idx[sampled_idx]
            out_ts[i, num_neighbors - n_sample:] = ngh_ts[sampled_idx]
            out_eidx[i, num_neighbors - n_sample:] = ngh_eidx[sampled_idx]

            # NEW: Mark which sampled neighbors match rules
            if matching_rels:
                for j, si in enumerate(sampled_idx):
                    if int(ngh_eidx[si]) in matching_rels:
                        out_rule_mask[i, num_neighbors - n_sample + j] = 1.0

        return out_node, out_eidx, out_ts, out_rule_mask
