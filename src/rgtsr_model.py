"""
RGTSR Model v3 - Rule-Aware Attention Architecture.

Key architectural change: Rule information is injected INSIDE the attention
mechanism, not just at the sampling stage.

Change 1: Rule-Aware Attention Score
  attention(vi, vj, rel) = original_attention + gamma * rule_prior(rel, hop)
  where rule_prior is a learned scalar indicating how well this edge's
  relation aligns with active rules at the current hop.

Change 2: Rule Prior as Edge Bias
  Each edge gets a prior score from rule matching. This prior is added
  to the attention logits BEFORE softmax, so it directly influences
  which edges get high attention weight.

Change 3: Per-query rule tracking through the full forward pass.
"""

import os, sys, time
from typing import List
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from segment import segment_softmax_op_v2, segment_norm_l1


class TimeEncode(nn.Module):
    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        super().__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific
        if entity_specific:
            self.basis_freq = nn.Parameter(torch.from_numpy(1/10**np.linspace(0,9,self.time_dim)).float().unsqueeze(0).repeat(num_entities,1))
            self.phase = nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(0).repeat(num_entities,1))
        else:
            self.basis_freq = nn.Parameter(torch.from_numpy(1/10**np.linspace(0,9,self.time_dim)).float())
            self.phase = nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        ts = ts.unsqueeze(2)
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(1) + self.phase[entities].unsqueeze(1)
        else:
            map_ts = ts * self.basis_freq.view(1,1,-1) + self.phase.view(1,1,-1)
        return torch.cos(map_ts)


class G3(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0/dim_in))
        self.key_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0/dim_in))

    def forward(self, inputs):
        vi, vj = inputs
        left_x = torch.cat(vi, dim=-1)
        right_x = torch.cat(vj, dim=-1)
        return torch.sum(self.query_proj(left_x) * self.key_proj(right_x), dim=-1)


class RuleAwareAttentionFlow(nn.Module):
    """
    Attention flow with rule-aware bias.

    KEY CHANGE vs original AttentionFlow:
      logits = G3(vi, vj, rel, query) + gamma * rule_prior
    where rule_prior is a per-edge scalar from rule matching.
    gamma is a LEARNED parameter that the model optimizes.
    """

    def __init__(self, n_dims_in, n_dims_out, ratio_update=0,
                 node_score_aggregation='sum', device='cpu'):
        super().__init__()
        self.transition_fn = G3(4 * n_dims_in, 4 * n_dims_in)
        self.linear_between_steps = nn.Linear(n_dims_in, n_dims_out, bias=True)
        nn.init.xavier_normal_(self.linear_between_steps.weight)
        self.act_between_steps = nn.LeakyReLU()

        # NEW: Learned rule bias weight - model learns how much to trust rules
        self.rule_gamma = nn.Parameter(torch.tensor(1.0))

        self.node_score_aggregation = node_score_aggregation
        self.ratio_update = ratio_update
        self.query_src_ts_emb = None
        self.query_rel_emb = None
        self.device = device

    def set_query_emb(self, qst, qr):
        self.query_src_ts_emb, self.query_rel_emb = qst, qr

    def _topk_att_score(self, edges, logits, k):
        res_e, res_l, res_i = [], [], []
        for eg in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg
            oi = np.arange(len(edges))[mask]
            me = edges[mask]
            ml = logits[mask]
            if me.shape[0] <= k:
                res_e.append(me); res_l.append(ml); res_i.append(oi)
            else:
                tl, ti = torch.topk(ml, k)
                res_i.append(oi[ti.cpu().numpy()])
                res_e.append(me[ti.cpu().numpy()])
                res_l.append(tl)
        return np.concatenate(res_e), torch.cat(res_l), np.concatenate(res_i)

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb):
        qst = torch.index_select(self.query_src_ts_emb, 0,
                                  torch.from_numpy(query_idx).long().to(self.device))
        qr = torch.index_select(self.query_rel_emb, 0,
                                 torch.from_numpy(query_idx).long().to(self.device))
        return self.transition_fn(((hidden_vi, rel_emb, qst, qr),
                                    (hidden_vj, rel_emb, qst, qr)))

    def forward(self, visited_node_score, selected_edges_l=None,
                visited_node_representation=None, rel_emb_l=None,
                rule_prior=None, max_edges=10, tc=None):
        """
        Forward with rule-aware attention.
        rule_prior: Tensor same length as selected_edges_l[-1], per-edge rule alignment score.
        """
        edges = selected_edges_l[-1]
        node_repr = visited_node_representation

        # Compute base attention logits
        hidden_vi = node_repr[edges[:, -2]]
        hidden_vj = node_repr[edges[:, -1]]
        base_logits = self.cal_attention_score(edges[:, 0], hidden_vi, hidden_vj, rel_emb_l[-1])

        # KEY CHANGE: Add rule prior bias to attention logits BEFORE softmax
        if rule_prior is not None and rule_prior.shape[0] == base_logits.shape[0]:
            logits = base_logits + self.rule_gamma * rule_prior
        else:
            logits = base_logits

        src_score = visited_node_score[edges[:, -2]]
        transition_softmax = segment_softmax_op_v2(logits, edges[:, -2], tc=tc)
        target_score = transition_softmax * src_score

        pruned_edges, pruned_target_score, orig_indices = self._topk_att_score(edges, target_score, max_edges)
        pruned_src_score = src_score[orig_indices]
        transition_pruned = transition_softmax[orig_indices]

        # Update node scores
        num_nodes = len(node_repr)
        sp_idx = torch.LongTensor(np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])).to(self.device)
        tm = torch.sparse.FloatTensor(sp_idx, transition_pruned, torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
        updated_node_score = torch.squeeze(torch.sparse.mm(tm, pruned_src_score.unsqueeze(1)))

        # Update node representations
        updated_repr = self._update_node_repr(pruned_edges, node_repr, transition_pruned, linear_act=False)

        # Backward through previous layers (same as xERTE)
        for se, re in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            h_vi = updated_repr[se[:, -2]]
            h_vj = updated_repr[se[:, -1]]
            tl = self.cal_attention_score(se[:, 0], h_vi, h_vj, re)
            ts = segment_softmax_op_v2(tl, se[:, -2], tc=tc)
            updated_repr = self._update_node_repr(se, updated_repr, ts, linear_act=False)

        updated_repr = self.bypass_forward(updated_repr)
        return updated_node_score, updated_repr, pruned_edges, orig_indices

    def _update_node_repr(self, edges, node_repr, transition_logits, linear_act=True):
        num_nodes = len(node_repr)
        sp_rep = torch.from_numpy(edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sv_rep = (1 - self.ratio_update) * transition_logits
        sp_id = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), edges[:, -2])).unsqueeze(1).repeat(1, 2).to(self.device)
        sv_id = torch.ones(len(sp_id)).to(self.device)
        sp_self = torch.from_numpy(np.unique(edges[:, -2])).unsqueeze(1).repeat(1, 2).to(self.device)
        sv_self = self.ratio_update * torch.ones(len(sp_self)).to(self.device)
        sp = torch.cat([sp_rep, sp_id, sp_self])
        sv = torch.cat([sv_rep, sv_id, sv_self])
        tm = torch.sparse.FloatTensor(sp.t(), sv, torch.Size([num_nodes, num_nodes])).to(self.device)
        updated = torch.sparse.mm(tm, node_repr)
        if linear_act:
            updated = self.act_between_steps(self.linear_between_steps(updated))
        return updated

    def bypass_forward(self, emb):
        return self.act_between_steps(self.linear_between_steps(emb))


class RGTSR(nn.Module):
    """
    RGTSR v3 with Rule-Aware Attention.

    Key differences from v2:
    1. RuleAwareAttentionFlow replaces AttentionFlow
    2. Rule prior signal flows through attention mechanism
    3. Per-edge rule_mask tracked from sampling through to attention
    """

    def __init__(self, ngh_finder, rules_dict=None, num_entity=None, num_rel=None,
                 emb_dim: List[int]=None, DP_num_edges=40, DP_steps=3,
                 alpha=0.5, emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', ent_score_aggregation='sum',
                 max_attended_edges=20, ratio_update=0,
                 device='cpu', use_time_embedding=True, **kwargs):
        super().__init__()
        assert len(emb_dim) == DP_steps + 1

        self.DP_num_edges = DP_num_edges
        self.DP_steps = DP_steps
        self.use_time_embedding = use_time_embedding
        self.ngh_finder = ngh_finder
        self.alpha = alpha
        self.rules_dict = rules_dict or {}

        self.temporal_embed_dim = [int(emb_dim[_]*2/(1+emb_static_ratio)) for _ in range(DP_steps)]
        self.static_embed_dim = [emb_dim[_]*2 - self.temporal_embed_dim[_] for _ in range(DP_steps)]

        self.entity_raw_embed = nn.Embedding(num_entity, self.static_embed_dim[0]).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = nn.Embedding(num_rel + 1, emb_dim[0]).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel

        # KEY CHANGE: Use RuleAwareAttentionFlow instead of plain AttentionFlow
        self.att_flow_list = nn.ModuleList([
            RuleAwareAttentionFlow(emb_dim[_], emb_dim[_+1],
                                    node_score_aggregation=node_score_aggregation,
                                    ratio_update=ratio_update, device=device)
            for _ in range(DP_steps)
        ])

        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2*emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])
        nn.init.xavier_normal_(self.node_emb_proj.weight)
        self.max_attended_edges = max_attended_edges

        if use_time_embedding:
            self.time_encoder = TimeEncode(self.temporal_embed_dim[0], entity_specific=diac_embed,
                                           num_entities=num_entity, device=device)
        self.ent_spec_time_embed = diac_embed
        self.device = device
        self.ent_score_aggregation = ent_score_aggregation

    def get_ent_emb(self, idx, device):
        ed = next(self.entity_raw_embed.parameters()).get_device()
        ed = torch.device('cpu') if ed == -1 else torch.device(f'cuda:{ed}')
        return self.entity_raw_embed(torch.from_numpy(idx).long().to(ed)).to(device)

    def get_rel_emb(self, idx, device):
        ed = next(self.relation_raw_embed.parameters()).get_device()
        ed = torch.device('cpu') if ed == -1 else torch.device(f'cuda:{ed}')
        return self.relation_raw_embed(torch.from_numpy(idx).long().to(ed)).to(device)

    def get_node_emb(self, src_idx_l, cut_time_l, eg_idx):
        hidden = self.get_ent_emb(src_idx_l, self.device)
        if self.use_time_embedding:
            ct = cut_time_l - self.cut_time_l[eg_idx]
            if self.ent_spec_time_embed:
                ht = self.time_encoder(torch.from_numpy(ct[:, np.newaxis]).to(self.device), entities=src_idx_l)
            else:
                ht = self.time_encoder(torch.from_numpy(ct[:, np.newaxis]).to(self.device))
            return self.node_emb_proj(torch.cat([hidden, ht.squeeze(1)], axis=1))
        return self.node_emb_proj(hidden)

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l):
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []
        self.rule_prior_l = []  # NEW: store rule priors per step
        self.node2index = {(i, src, ts): i for i, (src, _, ts) in
                           enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))}
        self.num_existing_nodes = len(src_idx_l)

        # Per-query rules
        self.batch_rules = {}
        for i, rel in enumerate(rel_idx_l):
            rules = self.rules_dict.get(int(rel), [])
            if rules:
                self.batch_rules[i] = rules

        # Query embeddings
        qsrc = self.get_ent_emb(src_idx_l, self.device)
        qrel = self.get_rel_emb(rel_idx_l, self.device)
        if self.use_time_embedding:
            if self.ent_spec_time_embed:
                qts = self.time_encoder(torch.zeros(len(cut_time_l),1).float().to(self.device), entities=src_idx_l)
            else:
                qts = self.time_encoder(torch.zeros(len(cut_time_l),1).float().to(self.device))
            qst = self.node_emb_proj(torch.cat([qsrc, qts.squeeze(1)], axis=1))
        else:
            qst = self.node_emb_proj(qsrc)

        for i, af in enumerate(self.att_flow_list):
            if i > 0:
                qst = self.att_flow_list[i-1].bypass_forward(qst)
                qrel = self.att_flow_list[i-1].bypass_forward(qrel)
            af.set_query_emb(qst, qrel)

    def initialize(self):
        eg = np.arange(len(self.src_idx_l), dtype=np.int32)
        att = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)
        attended = np.stack([eg, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l))], axis=1)
        return attended, attended.copy(), torch.from_numpy(att).to(self.device), self.att_flow_list[0].query_src_ts_emb

    def forward(self, sample):
        src_idx_l, rel_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.ts
        self.set_init(src_idx_l, rel_idx_l, cut_time_l)
        attended, visited, v_score, v_repr = self.initialize()

        for step in range(self.DP_steps):
            # Update rule matching for this hop
            self.ngh_finder.set_per_query_rules(self.batch_rules, current_hop=step)

            attended, visited, v_score, v_repr = \
                self._flow(attended, visited, v_score, v_repr, step)
            v_score = segment_norm_l1(v_score, visited[:, 0])

        entity_score, entities = self._get_entity_score(v_score[attended[:, -1]], attended)
        return entity_score, entities

    def _flow(self, attended, visited, v_score, v_repr, step, tc=None):
        # Sample edges with rule mask
        sampled_edges, new_nodes, new_attended, rule_prior = \
            self._get_sampled_edges(attended, self.DP_num_edges, step, add_self_loop=True)

        if len(new_nodes):
            new_emb = self.get_node_emb(new_nodes[:,1], new_nodes[:,2], eg_idx=new_nodes[:,0])
            for i in range(step):
                new_emb = self.att_flow_list[i].bypass_forward(new_emb)
            v_repr = torch.cat([v_repr, new_emb], axis=0)
            visited = np.concatenate([visited, new_nodes], axis=0)

        self.sampled_edges_l.append(sampled_edges)

        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        for i in range(step):
            rel_emb = self.att_flow_list[i].bypass_forward(rel_emb)
        for j in range(step):
            self.rel_emb_l[j] = self.att_flow_list[step-1].bypass_forward(self.rel_emb_l[j])
        self.rel_emb_l.append(rel_emb)
        self.rule_prior_l.append(rule_prior)

        # KEY CHANGE: Pass rule_prior to attention flow
        new_score, updated_repr, pruned_edges, orig_indices = \
            self.att_flow_list[step](v_score,
                                     selected_edges_l=self.sampled_edges_l,
                                     visited_node_representation=v_repr,
                                     rel_emb_l=self.rel_emb_l,
                                     rule_prior=rule_prior,
                                     max_edges=self.max_attended_edges, tc=tc)

        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]
        self.rule_prior_l[-1] = self.rule_prior_l[-1][orig_indices]

        _, idx = np.unique(pruned_edges[:, [0,4,3]], return_index=True, axis=0)
        updated_attended = pruned_edges[:, [0,3,4,7]][idx]
        return updated_attended, visited, new_score, updated_repr

    def _get_sampled_edges(self, attended, num_neighbors=20, step=None, add_self_loop=True):
        """Sample edges and return rule_prior tensor for attention bias."""
        src_idx = attended[:, 1]
        cut_time = attended[:, 2]
        node_idx = attended[:, 3]
        eg_idx = attended[:, 0]

        # Get neighbors WITH rule mask from neighbor finder
        ngh_node, ngh_eidx, ngh_ts, ngh_rule_mask = \
            self.ngh_finder.get_temporal_neighbor(src_idx, cut_time, eg_idx_l=eg_idx,
                                                   num_neighbors=num_neighbors)

        if add_self_loop:
            ngh_node = np.concatenate([ngh_node, src_idx[:, np.newaxis]], axis=1)
            ngh_eidx = np.concatenate([ngh_eidx, np.full((len(attended),1), self.selfloop, dtype=np.int32)], axis=1)
            ngh_ts = np.concatenate([ngh_ts, cut_time[:, np.newaxis]], axis=1)
            ngh_rule_mask = np.concatenate([ngh_rule_mask, np.zeros((len(attended),1), dtype=np.float32)], axis=1)

        n_per_node = num_neighbors + int(add_self_loop)
        flat_node = ngh_node.flatten()
        flat_eidx = ngh_eidx.flatten()
        flat_ts = ngh_ts.flatten()
        flat_rule = ngh_rule_mask.flatten()
        flat_eg = np.repeat(eg_idx, n_per_node)
        mask = flat_node != -1

        sampled_edges = np.stack([
            flat_eg, np.repeat(src_idx, n_per_node),
            np.repeat(cut_time, n_per_node),
            flat_node, flat_ts, flat_eidx,
            np.repeat(node_idx, n_per_node)
        ], axis=1)[mask]

        # Rule prior for each sampled edge
        rule_prior_np = flat_rule[mask]

        # Index new nodes
        target_idx = []
        new_nodes = []
        for eg, tn, tt in sampled_edges[:, [0, 3, 4]]:
            key = (eg, tn, tt)
            if key in self.node2index:
                target_idx.append(self.node2index[key])
            else:
                self.node2index[key] = self.num_existing_nodes
                target_idx.append(self.num_existing_nodes)
                new_nodes.append([eg, tn, tt, self.num_existing_nodes])
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate([sampled_edges, np.array(target_idx)[:, np.newaxis]], axis=1)
        new_nodes = sorted(new_nodes, key=lambda x: x[-1])
        new_nodes = np.array(new_nodes) if new_nodes else np.array([]).reshape(0, 4)

        _, att_idx = np.unique(sampled_edges[:, [0,4,3]], return_index=True, axis=0)
        new_attended = sampled_edges[:, [0,3,4]][att_idx]

        # Convert rule prior to tensor
        rule_prior_tensor = torch.from_numpy(rule_prior_np).float().to(self.device)

        return sampled_edges, new_nodes, new_attended, rule_prior_tensor

    def _get_entity_score(self, logits, nodes):
        device = logits.get_device()
        device = torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}')
        n = len(nodes)
        entities, eidx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
        sp_idx = torch.LongTensor(np.stack([eidx, np.arange(n)]))
        sp_val = torch.ones(n, dtype=torch.float)
        if self.ent_score_aggregation == 'mean':
            c = Counter([(nd[0], nd[1]) for nd in nodes[:, :2]])
            cnt = torch.tensor([c[(nd[0], nd[1])] for nd in nodes[:, :2]])
            sp_val = sp_val / cnt
        tm = torch.sparse.FloatTensor(sp_idx, sp_val, torch.Size([len(entities), n])).to(device)
        scores = torch.squeeze(torch.sparse.mm(tm, logits.unsqueeze(1)))
        return scores, entities

    def loss(self, entity_score, entities, target_idx_l, batch_size,
             gradient_iters_per_update=1, loss_fn='BCE'):
        label = torch.from_numpy(
            np.array([float(v == target_idx_l[eg]) for eg, v in entities], dtype=np.float32)
        ).to(self.device)
        score = entity_score * 0.999 + 0.0009
        if loss_fn == 'BCE':
            if gradient_iters_per_update == 1:
                return nn.BCELoss()(score, label)
            return nn.BCELoss(reduction='sum')(score, label) / (gradient_iters_per_update * batch_size)
        else:
            if gradient_iters_per_update == 1:
                return nn.NLLLoss()(score, label)
            return nn.NLLLoss(reduction='sum')(score, label) / (gradient_iters_per_update * batch_size)
