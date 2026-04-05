"""
Rule-Guided Neighbor Finder for RGTSR.

Extends xERTE's NeighborFinder with rule-aware sampling bias.
When sampling neighbors for subgraph expansion, edges whose relation
matches a body atom of an applicable rule get boosted probability.

Key formula:
  P(edge) ∝ exp((t_edge - t_query) / (granularity * weight_factor))
            * (1 + beta * rule_match(edge))

where rule_match(edge) = 1 if the edge's relation appears in the body
of a matched rule at the current hop position.
"""

import numpy as np
from collections import defaultdict


class RuleGuidedNeighborFinder:
    """
    NeighborFinder with rule-guided temporal sampling.

    Parameters:
        adj: dict, adj[entity_id] = list of (object, relation, timestamp)
        sampling: int, sampling strategy
            -1: full neighborhood
             0: uniform
             1: first N neighbors
             2: last N neighbors
             3: time-difference weighted (original xERTE)
             5: rule-guided time-weighted (NEW - default for RGTSR)
        max_time: int, maximum timestamp in data
        num_entities: int, total number of entities
        weight_factor: float, scale for time difference in exponential weighting
        time_granularity: int, 1 for YAGO, 24 for ICEWS
        rule_beta: float, boost factor for rule-matching edges (default 2.0)
    """

    def __init__(self, adj, sampling=5, max_time=366*24, num_entities=None,
                 weight_factor=1, time_granularity=24, rule_beta=2.0):
        self.time_granularity = time_granularity
        self.sampling = sampling
        self.weight_factor = weight_factor
        self.rule_beta = rule_beta

        # Active rules for current query (set by set_active_rules before each forward pass)
        self.active_rules = []
        self.current_hop = 0

        # Build offset structures (same as xERTE)
        node_idx_l, node_ts_l, edge_idx_l, off_set_l, off_set_t_l = \
            self._init_off_set(adj, max_time, num_entities)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.off_set_t_l = off_set_t_l

    def _init_off_set(self, adj, max_time, num_entities):
        """Initialize offset arrays for efficient neighbor lookup (same logic as xERTE)."""
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
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
                np.searchsorted(curr_ts, cut_time, 'left')
                for cut_time in range(0, max_time + 1, self.time_granularity)
            ])

        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def set_active_rules(self, rules, current_hop=0):
        """
        Set active rules for the current query.
        Called before each forward pass with rules matching the query relation.

        Parameters:
            rules: list of rule dicts, each with 'body_rels', 'conf', etc.
            current_hop: int, which hop in the rule body we are currently expanding
        """
        self.active_rules = rules if rules else []
        self.current_hop = current_hop

    def get_rule_matching_rels(self, hop=None):
        """
        Get the set of relations that appear at position `hop` in any active rule body.

        Parameters:
            hop: int, position in rule body (0-indexed). If None, use self.current_hop.

        Returns:
            set of relation ids that are "preferred" at this hop
        """
        if hop is None:
            hop = self.current_hop
        matching_rels = set()
        for rule in self.active_rules:
            body_rels = rule['body_rels']
            if hop < len(body_rels):
                matching_rels.add(body_rels[hop])
        return matching_rels

    def get_temporal_degree(self, src_idx_l, cut_time_l):
        """Return number of temporal neighbors for each (src, ts) pair."""
        temp_degree = []
        for src_idx, cut_time in zip(src_idx_l, cut_time_l):
            temp_degree.append(
                self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)]
            )
        return np.array(temp_degree)

    def find_before(self, src_idx, cut_time):
        """Find all neighbors of src_idx before cut_time."""
        neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        mid = np.searchsorted(neighbors_ts, cut_time)
        return neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Sample temporal neighbors with rule-guided bias.

        For sampling mode 5 (rule-guided), the sampling probability is:
          P(edge) ∝ exp((t - t_query) / scale) * (1 + beta * rule_match)

        where rule_match = 1 if edge relation is in the set of preferred
        relations at the current hop position.

        Parameters:
            src_idx_l: np.array, entity indices
            cut_time_l: np.array, cut-off timestamps
            num_neighbors: int, number of neighbors to sample

        Returns:
            out_ngh_node_batch: np.array (batch, num_neighbors), neighbor entity ids
            out_ngh_eidx_batch: np.array (batch, num_neighbors), neighbor relation ids
            out_ngh_t_batch: np.array (batch, num_neighbors), neighbor timestamps
        """
        assert len(src_idx_l) == len(cut_time_l)

        out_ngh_node_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_eidx_batch = -np.ones((len(src_idx_l), num_neighbors)).astype(np.int32)

        # Get rule-matching relations for current hop
        matching_rels = self.get_rule_matching_rels()

        if self.sampling == -1:
            full_ngh_node = []
            full_ngh_t = []
            full_ngh_edge = []

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]

            mid = self.off_set_t_l[src_idx][int(cut_time / self.time_granularity)]
            ngh_idx = neighbors_idx[:mid]
            ngh_eidx = neighbors_e_idx[:mid]
            ngh_ts = neighbors_ts[:mid]

            if len(ngh_idx) == 0:
                continue

            if self.sampling == 0:  # Uniform
                sampled_idx = np.sort(np.random.randint(0, len(ngh_idx), num_neighbors))
                out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

            elif self.sampling == 1:  # First N
                ngh_ts_sel = ngh_ts[:num_neighbors]
                ngh_idx_sel = ngh_idx[:num_neighbors]
                ngh_eidx_sel = ngh_eidx[:num_neighbors]
                out_ngh_node_batch[i, num_neighbors - len(ngh_idx_sel):] = ngh_idx_sel
                out_ngh_t_batch[i, num_neighbors - len(ngh_ts_sel):] = ngh_ts_sel
                out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx_sel):] = ngh_eidx_sel

            elif self.sampling == 2:  # Last N
                ngh_ts_sel = ngh_ts[-num_neighbors:]
                ngh_idx_sel = ngh_idx[-num_neighbors:]
                ngh_eidx_sel = ngh_eidx[-num_neighbors:]
                out_ngh_node_batch[i, num_neighbors - len(ngh_idx_sel):] = ngh_idx_sel
                out_ngh_t_batch[i, num_neighbors - len(ngh_ts_sel):] = ngh_ts_sel
                out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx_sel):] = ngh_eidx_sel

            elif self.sampling == 3:  # Time-weighted (original xERTE)
                delta_t = (ngh_ts - cut_time) / (self.time_granularity * self.weight_factor)
                weights = np.exp(delta_t) + 1e-9
                weights = weights / weights.sum()

                n_sample = min(len(ngh_idx), num_neighbors)
                sampled_idx = np.sort(
                    np.random.choice(len(ngh_idx), n_sample, replace=False, p=weights)
                )
                out_ngh_node_batch[i, num_neighbors - n_sample:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, num_neighbors - n_sample:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, num_neighbors - n_sample:] = ngh_eidx[sampled_idx]

            elif self.sampling == 5:  # Rule-guided time-weighted (NEW)
                delta_t = (ngh_ts - cut_time) / (self.time_granularity * self.weight_factor)
                time_weights = np.exp(delta_t) + 1e-9

                # Rule matching boost
                if matching_rels:
                    rule_boost = np.array([
                        1.0 + self.rule_beta if int(rel) in matching_rels else 1.0
                        for rel in ngh_eidx
                    ])
                else:
                    rule_boost = np.ones(len(ngh_eidx))

                weights = time_weights * rule_boost
                weights = weights / weights.sum()

                n_sample = min(len(ngh_idx), num_neighbors)
                sampled_idx = np.sort(
                    np.random.choice(len(ngh_idx), n_sample, replace=False, p=weights)
                )
                out_ngh_node_batch[i, num_neighbors - n_sample:] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, num_neighbors - n_sample:] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, num_neighbors - n_sample:] = ngh_eidx[sampled_idx]

            elif self.sampling == -1:  # Full neighborhood
                full_ngh_node.append(ngh_idx[-300:])
                full_ngh_t.append(ngh_ts[-300:])
                full_ngh_edge.append(ngh_eidx[-300:])

        if self.sampling == -1:
            if full_ngh_node:
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
