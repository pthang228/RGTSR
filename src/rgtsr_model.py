"""
RGTSR Model - Rule-Guided Temporal Subgraph Reasoning.

Core hybrid model combining:
  - TLogic's temporal random walks + rule learning (Phase 1)
  - Rule-guided subgraph construction (Phase 2 - NEW)
  - xERTE's temporal relational graph attention (Phase 3)
  - Combined scoring: α * attention_score + (1-α) * rule_confidence (Phase 4)

This file contains:
  - TimeEncode: Bochner's time embedding (from xERTE)
  - G3: Bilinear attention function (from xERTE)
  - AttentionFlow: Single attention propagation step (from xERTE)
  - RGTSR: Main model class (NEW - extends xERTE with rule guidance)
"""

import os
import sys
import time
from typing import List
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from segment import (segment_softmax_op_v2, segment_topk,
                     segment_norm_l1_part, segment_norm_l1)


# ===========================================================================
# TimeEncode - Bochner's time embedding (from xERTE, unchanged)
# ===========================================================================
class TimeEncode(nn.Module):
    """
    Bochner's time embedding.
    Maps scalar timestamps to high-dimensional representations using
    learned frequency and phase parameters.
    """

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific

        if entity_specific:
            self.basis_freq = nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float()
                .unsqueeze(0).repeat(num_entities, 1)
            )
            self.phase = nn.Parameter(
                torch.zeros(self.time_dim).float().unsqueeze(0).repeat(num_entities, 1)
            )
        else:
            self.basis_freq = nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float()
            )
            self.phase = nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2)

        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(dim=1)
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


# ===========================================================================
# G3 - Bilinear attention scoring function (from xERTE, unchanged)
# ===========================================================================
class G3(nn.Module):
    """
    Bilinear mapping for computing attention scores.
    score = MLP_1(x)^T * MLP_2(y) (element-wise product, then sum)
    """

    def __init__(self, dim_in, dim_out):
        super(G3, self).__init__()
        self.dim_out = dim_out
        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0 / dim_in))
        self.key_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0 / dim_in))

    def forward(self, inputs):
        vi, vj = inputs
        left_x = torch.cat(vi, dim=-1)
        right_x = torch.cat(vj, dim=-1)
        return torch.sum(self.query_proj(left_x) * self.key_proj(right_x), dim=-1)


# ===========================================================================
# AttentionFlow - Single attention propagation step (from xERTE, unchanged)
# ===========================================================================
class AttentionFlow(nn.Module):
    """
    One step of attention-based message passing and score propagation.
    Computes transition attention, prunes edges, updates node representations.
    """

    def __init__(self, n_dims_in, n_dims_out, ratio_update=0,
                 node_score_aggregation='sum', device='cpu'):
        super(AttentionFlow, self).__init__()
        self.transition_fn = G3(4 * n_dims_in, 4 * n_dims_in)
        self.linear_between_steps = nn.Linear(n_dims_in, n_dims_out, bias=True)
        nn.init.xavier_normal_(self.linear_between_steps.weight)
        self.act_between_steps = nn.LeakyReLU()
        self.node_score_aggregation = node_score_aggregation
        self.ratio_update = ratio_update
        self.query_src_ts_emb = None
        self.query_rel_emb = None
        self.device = device

    def set_query_emb(self, query_src_ts_emb, query_rel_emb):
        self.query_src_ts_emb = query_src_ts_emb
        self.query_rel_emb = query_rel_emb

    def _topk_att_score(self, edges, logits, k, tc=None):
        """Prune edges keeping top-k per query subgraph."""
        res_edges = []
        res_logits = []
        res_indices = []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_logits = logits[mask]
            if masked_edges.shape[0] <= k:
                res_edges.append(masked_edges)
                res_logits.append(masked_logits)
                res_indices.append(orig_indices)
            else:
                topk_logits, indices = torch.topk(masked_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                res_edges.append(masked_edges[indices.cpu().numpy()])
                res_logits.append(topk_logits)

        return (np.concatenate(res_edges, axis=0),
                torch.cat(res_logits, dim=0),
                np.concatenate(res_indices, axis=0))

    def _cal_attention_score(self, edges, memorized_embedding, rel_emb):
        hidden_vi = memorized_embedding[edges[:, -2]]
        hidden_vj = memorized_embedding[edges[:, -1]]
        return self.cal_attention_score(edges[:, 0], hidden_vi, hidden_vj, rel_emb)

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb):
        query_src_ts_emb_repeat = torch.index_select(
            self.query_src_ts_emb, dim=0,
            index=torch.from_numpy(query_idx).long().to(self.device)
        )
        query_rel_emb_repeat = torch.index_select(
            self.query_rel_emb, dim=0,
            index=torch.from_numpy(query_idx).long().to(self.device)
        )
        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat),
             (hidden_vj, rel_emb, query_src_ts_emb_repeat, query_rel_emb_repeat))
        )
        return transition_logits

    def forward(self, visited_node_score, selected_edges_l=None,
                visited_node_representation=None, rel_emb_l=None,
                max_edges=10, analysis=False, tc=None):
        """Forward pass: compute attention, prune, propagate scores and representations."""

        transition_logits = self._cal_attention_score(
            selected_edges_l[-1], visited_node_representation, rel_emb_l[-1]
        )

        src_score = visited_node_score[selected_edges_l[-1][:, -2]]
        transition_logits_softmax = segment_softmax_op_v2(
            transition_logits, selected_edges_l[-1][:, -2], tc=tc
        )
        target_score = transition_logits_softmax * src_score

        pruned_edges, pruned_target_score, orig_indices = self._topk_att_score(
            selected_edges_l[-1], target_score, max_edges
        )
        pruned_src_score = src_score[orig_indices]
        transition_logits_pruned_softmax = transition_logits_softmax[orig_indices]

        num_nodes = len(visited_node_representation)
        if self.node_score_aggregation == 'sum':
            sparse_index = torch.LongTensor(
                np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])
            ).to(self.device)
            trans_matrix = torch.sparse.FloatTensor(
                sparse_index, transition_logits_pruned_softmax,
                torch.Size([num_nodes, len(pruned_edges)])
            ).to(self.device)
            updated_node_score = torch.squeeze(
                torch.sparse.mm(trans_matrix, pruned_src_score.unsqueeze(1))
            )
        elif self.node_score_aggregation == 'mean':
            sparse_index = torch.LongTensor(
                np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])
            ).to(self.device)
            c = Counter(pruned_edges[:, -1])
            target_node_cnt = torch.tensor([c[_] for _ in pruned_edges[:, -1]]).to(self.device)
            norm_logits = torch.div(transition_logits_pruned_softmax, target_node_cnt)
            trans_matrix = torch.sparse.FloatTensor(
                sparse_index, norm_logits,
                torch.Size([num_nodes, len(pruned_edges)])
            ).to(self.device)
            updated_node_score = torch.squeeze(
                torch.sparse.mm(trans_matrix, pruned_src_score.unsqueeze(1))
            )
        elif self.node_score_aggregation == 'max':
            max_dict = {}
            for i in range(len(pruned_edges)):
                score_i = pruned_target_score[i].cpu().detach().numpy()
                if score_i > max_dict.get(pruned_edges[i, -1], (0, 0))[1]:
                    max_dict[pruned_edges[i, -1]] = (i, score_i)
            sparse_index = torch.LongTensor(
                np.stack([np.array(list(max_dict.keys())),
                          np.array([_[0] for _ in max_dict.values()])])
            ).to(self.device)
            trans_matrix = torch.sparse.FloatTensor(
                sparse_index, torch.ones(len(max_dict)).to(self.device),
                torch.Size([num_nodes, len(pruned_edges)])
            ).to(self.device)
            updated_node_score = torch.squeeze(
                torch.sparse.mm(trans_matrix, pruned_target_score.unsqueeze(1))
            )
        else:
            raise ValueError("node_score_aggregation must be sum, mean, or max")

        updated_repr = self._update_node_representation_along_edges(
            pruned_edges, visited_node_representation,
            transition_logits_pruned_softmax, linear_act=False
        )

        # Backward propagation through previous layers
        for selected_edges, rel_emb in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            trans_logits = self._cal_attention_score(selected_edges, updated_repr, rel_emb)
            trans_softmax = segment_softmax_op_v2(trans_logits, selected_edges[:, -2], tc=tc)
            updated_repr = self._update_node_representation_along_edges(
                selected_edges, updated_repr, trans_softmax, linear_act=False
            )

        updated_repr = self.bypass_forward(updated_repr)

        if analysis:
            return (updated_node_score, updated_repr, pruned_edges, orig_indices,
                    transition_logits_softmax, [transition_logits_pruned_softmax])
        else:
            return updated_node_score, updated_repr, pruned_edges, orig_indices

    def _update_node_representation_along_edges(self, edges, node_representation,
                                                 transition_logits, linear_act=True):
        num_nodes = len(node_representation)
        sparse_index_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_value_rep = (1 - self.ratio_update) * transition_logits

        sparse_index_identical = torch.from_numpy(
            np.setdiff1d(np.arange(num_nodes), edges[:, -2])
        ).unsqueeze(1).repeat(1, 2).to(self.device)
        sparse_value_identical = torch.ones(len(sparse_index_identical)).to(self.device)

        sparse_index_self = torch.from_numpy(
            np.unique(edges[:, -2])
        ).unsqueeze(1).repeat(1, 2).to(self.device)
        sparse_value_self = self.ratio_update * torch.ones(len(sparse_index_self)).to(self.device)

        sparse_index = torch.cat([sparse_index_rep, sparse_index_identical, sparse_index_self], axis=0)
        sparse_value = torch.cat([sparse_value_rep, sparse_value_identical, sparse_value_self])
        trans_matrix = torch.sparse.FloatTensor(
            sparse_index.t(), sparse_value,
            torch.Size([num_nodes, num_nodes])
        ).to(self.device)
        updated = torch.sparse.mm(trans_matrix, node_representation)
        if linear_act:
            updated = self.act_between_steps(self.linear_between_steps(updated))
        return updated

    def bypass_forward(self, embedding):
        return self.act_between_steps(self.linear_between_steps(embedding))


# ===========================================================================
# RGTSR - Main hybrid model
# ===========================================================================
class RGTSR(nn.Module):
    """
    Rule-Guided Temporal Subgraph Reasoning model.

    Extends xERTE architecture with:
      1. Rule-guided neighbor sampling (via RuleGuidedNeighborFinder)
      2. Per-step rule hop tracking (expansion follows rule body structure)
      3. Combined scoring: α * neural_attention + (1-α) * rule_confidence
      4. Inductive fallback to pure rule-based scoring for unseen entities

    Parameters:
        ngh_finder: RuleGuidedNeighborFinder instance
        rules_dict: dict, relation_id -> list of rules (from TLogic Phase 1)
        num_entity: int
        num_rel: int
        emb_dim: list of ints, embedding dimensions per step
        DP_num_edges: int, number of edges sampled per node per step
        DP_steps: int, number of attention propagation steps
        alpha: float, weight for neural vs rule score combination
        emb_static_ratio: float, ratio of static to temporal embedding
        diac_embed: bool, entity-specific time embeddings
        node_score_aggregation: str, 'sum', 'mean', or 'max'
        ent_score_aggregation: str, 'sum' or 'mean'
        max_attended_edges: int, max edges after pruning
        ratio_update: float, self vs neighbor update ratio
        device: str, 'cpu' or 'cuda:X'
        use_time_embedding: bool
    """

    def __init__(self, ngh_finder, rules_dict=None, num_entity=None, num_rel=None,
                 emb_dim: List[int] = None, DP_num_edges=40, DP_steps=3,
                 alpha=0.5, emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', ent_score_aggregation='sum',
                 max_attended_edges=20, ratio_update=0,
                 device='cpu', analysis=False, use_time_embedding=True, **kwargs):
        super(RGTSR, self).__init__()
        assert len(emb_dim) == DP_steps + 1

        self.DP_num_edges = DP_num_edges
        self.DP_steps = DP_steps
        self.use_time_embedding = use_time_embedding
        self.ngh_finder = ngh_finder
        self.alpha = alpha

        # Rules dict: relation_id -> sorted list of rules by confidence
        self.rules_dict = rules_dict if rules_dict is not None else {}

        # Embedding dimensions
        self.temporal_embed_dim = [int(emb_dim[_] * 2 / (1 + emb_static_ratio)) for _ in range(DP_steps)]
        self.static_embed_dim = [emb_dim[_] * 2 - self.temporal_embed_dim[_] for _ in range(DP_steps)]

        # Entity and relation embeddings (kept on CPU for memory, same as xERTE)
        self.entity_raw_embed = nn.Embedding(num_entity, self.static_embed_dim[0]).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = nn.Embedding(num_rel + 1, emb_dim[0]).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel

        # Attention flow layers
        self.att_flow_list = nn.ModuleList([
            AttentionFlow(emb_dim[_], emb_dim[_ + 1],
                          node_score_aggregation=node_score_aggregation,
                          ratio_update=ratio_update, device=device)
            for _ in range(DP_steps)
        ])

        # Projection layers
        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2 * emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])
        nn.init.xavier_normal_(self.node_emb_proj.weight)

        self.max_attended_edges = max_attended_edges

        # Time encoder
        if use_time_embedding:
            self.time_encoder = TimeEncode(
                expand_dim=self.temporal_embed_dim[0],
                entity_specific=diac_embed,
                num_entities=num_entity, device=device
            )
        self.ent_spec_time_embed = diac_embed

        self.device = device
        self.analysis = analysis
        self.ent_score_aggregation = ent_score_aggregation

    def load_rules(self, rules_dict):
        """Load rules learned from TLogic Phase 1."""
        self.rules_dict = rules_dict

    def _get_query_rules(self, rel_idx):
        """
        Get applicable rules for a given query relation.

        Parameters:
            rel_idx: int, query relation index

        Returns:
            list of rule dicts sorted by confidence (descending)
        """
        return self.rules_dict.get(int(rel_idx), [])

    def _get_max_rule_length(self, rules):
        """Get maximum body length among active rules."""
        if not rules:
            return self.DP_steps
        return max(len(r['body_rels']) for r in rules)

    def get_ent_emb(self, ent_idx_l, device):
        embed_device = next(self.entity_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.entity_raw_embed(
            torch.from_numpy(ent_idx_l).long().to(embed_device)
        ).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        embed_device = next(self.relation_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.relation_raw_embed(
            torch.from_numpy(rel_idx_l).long().to(embed_device)
        ).to(device)

    def get_node_emb(self, src_idx_l, cut_time_l, eg_idx):
        hidden_node = self.get_ent_emb(src_idx_l, self.device)
        if self.use_time_embedding:
            cut_time_l = cut_time_l - self.cut_time_l[eg_idx]
            if self.ent_spec_time_embed:
                hidden_time = self.time_encoder(
                    torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device),
                    entities=src_idx_l
                )
            else:
                hidden_time = self.time_encoder(
                    torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device)
                )
            return self.node_emb_proj(
                torch.cat([hidden_node, torch.squeeze(hidden_time, 1)], axis=1)
            )
        else:
            return self.node_emb_proj(hidden_node)

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l):
        """Initialize query state and set rule-guided neighbor sampling."""
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []

        self.node2index = {
            (i, src, ts): i
            for i, (src, rel, ts) in enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))
        }
        self.num_existing_nodes = len(src_idx_l)

        # Collect all applicable rules for queries in this batch
        self.batch_rules = {}
        for i, rel in enumerate(rel_idx_l):
            rules = self._get_query_rules(rel)
            if rules:
                self.batch_rules[i] = rules

        # Set active rules on the neighbor finder (union of all batch rules)
        all_rules = []
        for rules in self.batch_rules.values():
            all_rules.extend(rules)
        self.ngh_finder.set_active_rules(all_rules, current_hop=0)

        # Compute query embeddings
        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)

        if self.use_time_embedding:
            if self.ent_spec_time_embed:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).float().to(self.device),
                    entities=self.src_idx_l
                )
            else:
                query_ts_emb = self.time_encoder(
                    torch.zeros(len(self.cut_time_l), 1).float().to(self.device)
                )
            query_ts_emb = torch.squeeze(query_ts_emb, 1)
            query_src_ts_emb = self.node_emb_proj(
                torch.cat([query_src_emb, query_ts_emb], axis=1)
            )
        else:
            query_src_ts_emb = self.node_emb_proj(query_src_emb)

        for i, att_flow in enumerate(self.att_flow_list):
            if i > 0:
                query_src_ts_emb = self.att_flow_list[i - 1].bypass_forward(query_src_ts_emb)
                query_rel_emb = self.att_flow_list[i - 1].bypass_forward(query_rel_emb)
            att_flow.set_query_emb(query_src_ts_emb, query_rel_emb)

    def initialize(self):
        """Initialize node scores and representations."""
        eg_idx_l = np.arange(len(self.src_idx_l), dtype=np.int32)
        att_score = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)
        attended_nodes = np.stack(
            [eg_idx_l, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l))],
            axis=1
        )
        visited_nodes_score = torch.from_numpy(att_score).to(self.device)
        visited_nodes = attended_nodes
        visited_node_representation = self.att_flow_list[0].query_src_ts_emb
        return attended_nodes, visited_nodes, visited_nodes_score, visited_node_representation

    def forward(self, sample):
        """
        Forward pass with rule-guided subgraph expansion.

        At each DP step, the neighbor finder samples edges biased toward
        relations matching the rule body at the current hop position.
        """
        src_idx_l, rel_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.ts
        self.set_init(src_idx_l, rel_idx_l, cut_time_l)
        attended_nodes, visited_nodes, visited_node_score, visited_node_representation = \
            self.initialize()

        for step in range(self.DP_steps):
            # Update neighbor finder with current hop for rule-guided sampling
            self.ngh_finder.current_hop = step

            attended_nodes, visited_nodes, visited_node_score, visited_node_representation = \
                self._flow(attended_nodes, visited_nodes, visited_node_score,
                           visited_node_representation, step)
            visited_node_score = segment_norm_l1(visited_node_score, visited_nodes[:, 0])

        entity_att_score, entities = self.get_entity_attn_score(
            visited_node_score[attended_nodes[:, -1]], attended_nodes
        )
        return entity_att_score, entities

    def _flow(self, attended_nodes, visited_nodes, visited_node_score,
              visited_node_representation, step, tc=None):
        """Single expansion + attention step (same structure as xERTE)."""
        sampled_edges, new_sampled_nodes, new_attended_nodes = self._get_sampled_edges(
            attended_nodes, num_neighbors=self.DP_num_edges,
            step=step, add_self_loop=True, tc=tc
        )

        if len(new_sampled_nodes):
            new_emb = self.get_node_emb(
                new_sampled_nodes[:, 1], new_sampled_nodes[:, 2],
                eg_idx=new_sampled_nodes[:, 0]
            )
            for i in range(step):
                new_emb = self.att_flow_list[i].bypass_forward(new_emb)
            visited_node_representation = torch.cat(
                [visited_node_representation, new_emb], axis=0
            )
            visited_nodes = np.concatenate([visited_nodes, new_sampled_nodes], axis=0)

        self.sampled_edges_l.append(sampled_edges)

        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        for i in range(step):
            rel_emb = self.att_flow_list[i].bypass_forward(rel_emb)
        for j in range(step):
            self.rel_emb_l[j] = self.att_flow_list[step - 1].bypass_forward(self.rel_emb_l[j])
        self.rel_emb_l.append(rel_emb)

        new_score, updated_repr, pruned_edges, orig_indices = \
            self.att_flow_list[step](
                visited_node_score,
                selected_edges_l=self.sampled_edges_l,
                visited_node_representation=visited_node_representation,
                rel_emb_l=self.rel_emb_l,
                max_edges=self.max_attended_edges, tc=tc
            )

        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]

        _, indices = np.unique(pruned_edges[:, [0, 4, 3]], return_index=True, axis=0)
        updated_attended = pruned_edges[:, [0, 3, 4, 7]][indices]

        return updated_attended, visited_nodes, new_score, updated_repr

    def _get_sampled_edges(self, attended_nodes, num_neighbors=20,
                            step=None, add_self_loop=True, tc=None):
        """Sample neighborhood edges using rule-guided neighbor finder."""
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        node_idx_l = attended_nodes[:, 3]

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = \
            self.ngh_finder.get_temporal_neighbor(src_idx_l, cut_time_l,
                                                   num_neighbors=num_neighbors)

        # Add selfloop
        if add_self_loop:
            src_ngh_node_batch = np.concatenate(
                [src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1
            )
            src_ngh_eidx_batch = np.concatenate(
                [src_ngh_eidx_batch,
                 np.full((len(attended_nodes), 1), self.selfloop, dtype=np.int32)],
                axis=1
            )
            src_ngh_t_batch = np.concatenate(
                [src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1
            )

        # Flatten and filter padded entries
        src_ngh_node_flatten = src_ngh_node_batch.flatten()
        src_ngh_eidx_flatten = src_ngh_eidx_batch.flatten()
        src_ngh_t_flatten = src_ngh_t_batch.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + int(add_self_loop))
        mask = src_ngh_node_flatten != -1

        sampled_edges = np.stack([
            eg_idx,
            np.repeat(src_idx_l, num_neighbors + int(add_self_loop)),
            np.repeat(cut_time_l, num_neighbors + int(add_self_loop)),
            src_ngh_node_flatten,
            src_ngh_t_flatten,
            src_ngh_eidx_flatten,
            np.repeat(node_idx_l, num_neighbors + int(add_self_loop))
        ], axis=1)[mask]

        # Index new nodes
        target_nodes_index = []
        new_sampled_nodes = []
        for eg, tar_node, tar_ts in sampled_edges[:, [0, 3, 4]]:
            key = (eg, tar_node, tar_ts)
            if key in self.node2index:
                target_nodes_index.append(self.node2index[key])
            else:
                self.node2index[key] = self.num_existing_nodes
                target_nodes_index.append(self.num_existing_nodes)
                new_sampled_nodes.append([eg, tar_node, tar_ts, self.num_existing_nodes])
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate(
            [sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1
        )
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes) if new_sampled_nodes else np.array([]).reshape(0, 4)

        _, new_attended_idx = np.unique(sampled_edges[:, [0, 4, 3]], return_index=True, axis=0)
        new_attended_nodes = sampled_edges[:, [0, 3, 4]][new_attended_idx]

        return sampled_edges, new_sampled_nodes, new_attended_nodes

    def get_entity_attn_score(self, logits, nodes, tc=None):
        """Aggregate node scores to entity-level scores."""
        entity_attn_score, entities = self._aggregate_op_entity(
            logits, nodes, self.ent_score_aggregation
        )
        return entity_attn_score, entities

    def _aggregate_op_entity(self, logits, nodes, aggr='sum'):
        """Aggregate attention scores for the same entity across different timestamps."""
        device = logits.get_device()
        if device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        num_nodes = len(nodes)
        entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
        sparse_index = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)]))
        sparse_value = torch.ones(num_nodes, dtype=torch.float)

        if aggr == 'mean':
            c = Counter([(n[0], n[1]) for n in nodes[:, :2]])
            target_cnt = torch.tensor([c[(_[0], _[1])] for _ in nodes[:, :2]])
            sparse_value = torch.div(sparse_value, target_cnt)

        trans_matrix = torch.sparse.FloatTensor(
            sparse_index, sparse_value,
            torch.Size([len(entities), num_nodes])
        ).to(device)
        entity_att_score = torch.squeeze(
            torch.sparse.mm(trans_matrix, logits.unsqueeze(1))
        )
        return entity_att_score, entities

    def combined_score(self, entity_att_score, entities, rel_idx_l, alpha=None):
        """
        Combine neural attention scores with rule confidence scores.

        score(entity) = α * attention_score + (1-α) * rule_confidence_score

        Parameters:
            entity_att_score: Tensor, neural attention scores per entity
            entities: np.array (N, 2), (eg_idx, entity_id)
            rel_idx_l: np.array, query relation for each example in batch
            alpha: float, override self.alpha if provided

        Returns:
            combined_scores: Tensor, combined scores
        """
        if alpha is None:
            alpha = self.alpha

        # If no rules, return pure neural score
        if not self.rules_dict:
            return entity_att_score

        # Compute rule-based confidence for each candidate entity
        rule_scores = torch.zeros_like(entity_att_score)

        for i, (eg_idx, ent_id) in enumerate(entities):
            eg_idx = int(eg_idx)
            if eg_idx < len(rel_idx_l):
                rel = int(rel_idx_l[eg_idx])
                rules = self.rules_dict.get(rel, [])
                if rules:
                    # Use max rule confidence as rule score
                    max_conf = max(r['conf'] for r in rules)
                    rule_scores[i] = max_conf

        # Normalize rule scores per query
        rule_scores = segment_norm_l1(rule_scores, entities[:, 0])

        # Combine
        combined = alpha * entity_att_score + (1 - alpha) * rule_scores
        return combined

    def loss(self, entity_att_score, entities, target_idx_l, batch_size,
             gradient_iters_per_update=1, loss_fn='BCE'):
        """Compute loss (same as xERTE)."""
        one_hot_label = torch.from_numpy(
            np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities],
                     dtype=np.float32)
        ).to(self.device)

        entity_att_score = entity_att_score * 0.999 + 0.0009

        if loss_fn == 'BCE':
            if gradient_iters_per_update == 1:
                loss = nn.BCELoss()(entity_att_score, one_hot_label)
            else:
                loss = nn.BCELoss(reduction='sum')(entity_att_score, one_hot_label)
                loss /= gradient_iters_per_update * batch_size
        else:
            if gradient_iters_per_update == 1:
                loss = nn.NLLLoss()(entity_att_score, one_hot_label)
            else:
                loss = nn.NLLLoss(reduction='sum')(entity_att_score, one_hot_label)
                loss /= gradient_iters_per_update * batch_size

        return loss
