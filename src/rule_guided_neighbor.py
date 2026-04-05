"""
Rule-Guided Neighbor Finder for RGTSR (FIXED v2).

FIX 1: Per-query rule matching instead of batch-level union.
Each attended node belongs to a specific query (eg_idx), so the rule
boost is computed per-query, not as a global union across the batch.
"""

import numpy as np
from collections import defaultdict


class RuleGuidedNeighborFinder:
    def __init__(self, adj, sampling=5, max_time=366*24, num_entities=None,
                 weight_factor=1, time_granularity=24, rule_beta=2.0):
        self.time_granularity = time_granularity
        self.sampling = sampling
        self.weight_factor = weight_factor
        self.rule_beta = rule_beta
        self.current_hop = 0

        # FIX: Per-query rule matching instead of global
        self.per_query_matching_rels = {}

        node_idx_l, node_ts_l, edge_idx_l, off_set_l, off_set_t_l = \
            self._init_off_set(adj, max_time, num_entities)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.off_set_t_l = off_set_t_l

    def _init_off_set(self, adj, max_time, num_entities):
        n_idx_l, n_ts_l, e_idx_l = [], [], []
        off_set_l = [0]
        off_set_t_l = []
        for i in range(num_entities):
            curr = adj.get(i, [])
            curr = sorted(curr, key=lambda x: (int(x[2]), int(x[0]), int(x[1])))
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            curr_ts = [x[2] for x in curr]
            n_ts_l.extend(curr_ts)
            off_set_l.append(len(n_idx_l))
            off_set_t_l.append([
                np.searchsorted(curr_ts, ct, 'left')
                for ct in range(0, max_time + 1, self.time_granularity)
            ])
        return np.array(n_idx_l), np.array(n_ts_l), np.array(e_idx_l), np.array(off_set_l), off_set_t_l

    def set_per_query_rules(self, batch_rules, current_hop=0):
        """
        FIX: Set rules PER QUERY instead of a global union.

        Parameters:
            batch_rules: dict, eg_idx -> list of rules for that query
            current_hop: int, which hop we are expanding
        """
        self.current_hop = current_hop
        self.per_query_matching_rels = {}
        for eg_idx, rules in batch_rules.items():
            matching = set()
            for rule in rules:
                body_rels = rule['body_rels']
                if current_hop < len(body_rels):
                    matching.add(body_rels[current_hop])
            if matching:
                self.per_query_matching_rels[eg_idx] = matching

    def get_temporal_degree(self, src_idx_l, cut_time_l):
        temp_degree = []
        for src_idx, cut_time in zip(src_idx_l, cut_time_l):
            temp_degree.append(self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)])
        return np.array(temp_degree)

    def find_before(self, src_idx, cut_time):
        nidx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        nts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neidx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        mid = np.searchsorted(nts, cut_time)
        return nidx[:mid], neidx[:mid], nts[:mid]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, eg_idx_l=None, num_neighbors=20):
        """
        FIX: Now accepts eg_idx_l for per-query rule matching.
        """
        assert len(src_idx_l) == len(cut_time_l)

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        if self.sampling == -1:
            full_ngh_node, full_ngh_t, full_ngh_edge = [], [], []

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            nidx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            nts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neidx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)]
            ngh_idx, ngh_eidx, ngh_ts = nidx[:mid], neidx[:mid], nts[:mid]

            if len(ngh_idx) == 0:
                continue

            if self.sampling == 3:  # Time-weighted (xERTE baseline)
                delta_t = (ngh_ts - cut_time) / (self.time_granularity * self.weight_factor)
                weights = np.exp(delta_t) + 1e-9
                weights /= weights.sum()
                n_sample = min(len(ngh_idx), num_neighbors)
                sampled_idx = np.sort(np.random.choice(len(ngh_idx), n_sample, replace=False, p=weights))
                out_ngh_node_batch[i, num_neighbors - n_sample:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, num_neighbors - n_sample:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, num_neighbors - n_sample:] = ngh_eidx[sampled_idx]

            elif self.sampling == 5:  # Rule-guided (FIXED: per-query)
                delta_t = (ngh_ts - cut_time) / (self.time_granularity * self.weight_factor)
                time_weights = np.exp(delta_t) + 1e-9

                # FIX: Get matching rels for THIS specific query
                matching_rels = set()
                if eg_idx_l is not None:
                    eg = int(eg_idx_l[i])
                    matching_rels = self.per_query_matching_rels.get(eg, set())

                if matching_rels:
                    rule_boost = np.array([
                        1.0 + self.rule_beta if int(rel) in matching_rels else 1.0
                        for rel in ngh_eidx
                    ])
                else:
                    rule_boost = np.ones(len(ngh_eidx))

                weights = time_weights * rule_boost
                weights /= weights.sum()
                n_sample = min(len(ngh_idx), num_neighbors)
                sampled_idx = np.sort(np.random.choice(len(ngh_idx), n_sample, replace=False, p=weights))
                out_ngh_node_batch[i, num_neighbors - n_sample:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, num_neighbors - n_sample:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, num_neighbors - n_sample:] = ngh_eidx[sampled_idx]

            elif self.sampling == -1:
                full_ngh_node.append(ngh_idx[-300:])
                full_ngh_t.append(ngh_ts[-300:])
                full_ngh_edge.append(ngh_eidx[-300:])

        if self.sampling == -1 and full_ngh_node:
            max_num = max(len(x) for x in full_ngh_node)
            out_ngh_node_batch = -np.ones((len(src_idx_l), max_num)).astype(np.int32)
            out_ngh_t_batch = np.zeros((len(src_idx_l), max_num)).astype(np.int32)
            out_ngh_eidx_batch = -np.ones((len(src_idx_l), max_num)).astype(np.int32)
            for i in range(len(full_ngh_node)):
                n = len(full_ngh_node[i])
                out_ngh_node_batch[i, max_num - n:] = full_ngh_node[i]
                out_ngh_eidx_batch[i, max_num - n:] = full_ngh_edge[i]
                out_ngh_t_batch[i, max_num - n:] = full_ngh_t[i]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch
