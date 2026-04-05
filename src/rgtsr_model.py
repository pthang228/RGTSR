"""
RGTSR Model v2 (FIXED).

FIX 1: Per-query rule matching in neighbor finder (not batch union)
FIX 2: Combined scoring uses actual rule APPLICATION per candidate entity
FIX 3: Pass eg_idx to neighbor finder for per-query rule lookup
"""

import os, sys, time
from typing import List
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from segment import segment_softmax_op_v2, segment_topk, segment_norm_l1_part, segment_norm_l1


class TimeEncode(nn.Module):
    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        super().__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific
        if entity_specific:
            self.basis_freq = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(0).repeat(num_entities, 1))
            self.phase = nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(0).repeat(num_entities, 1))
        else:
            self.basis_freq = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())
            self.phase = nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities=None):
        ts = torch.unsqueeze(ts, dim=2)
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(dim=1) + self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1) + self.phase.view(1, 1, -1)
        return torch.cos(map_ts)


class G3(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.query_proj.weight, mean=0, std=np.sqrt(2.0 / dim_in))
        self.key_proj = nn.Linear(dim_in, dim_out, bias=False)
        nn.init.normal_(self.key_proj.weight, mean=0, std=np.sqrt(2.0 / dim_in))

    def forward(self, inputs):
        vi, vj = inputs
        left_x = torch.cat(vi, dim=-1)
        right_x = torch.cat(vj, dim=-1)
        return torch.sum(self.query_proj(left_x) * self.key_proj(right_x), dim=-1)


class AttentionFlow(nn.Module):
    def __init__(self, n_dims_in, n_dims_out, ratio_update=0, node_score_aggregation='sum', device='cpu'):
        super().__init__()
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
        self.query_src_ts_emb, self.query_rel_emb = query_src_ts_emb, query_rel_emb

    def _topk_att_score(self, edges, logits, k, tc=None):
        res_edges, res_logits, res_indices = [], [], []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_logits = logits[mask]
            if masked_edges.shape[0] <= k:
                res_edges.append(masked_edges); res_logits.append(masked_logits); res_indices.append(orig_indices)
            else:
                topk_logits, indices = torch.topk(masked_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                res_edges.append(masked_edges[indices.cpu().numpy()])
                res_logits.append(topk_logits)
        return np.concatenate(res_edges), torch.cat(res_logits), np.concatenate(res_indices)

    def _cal_attention_score(self, edges, memorized_embedding, rel_emb):
        hidden_vi = memorized_embedding[edges[:, -2]]
        hidden_vj = memorized_embedding[edges[:, -1]]
        return self.cal_attention_score(edges[:, 0], hidden_vi, hidden_vj, rel_emb)

    def cal_attention_score(self, query_idx, hidden_vi, hidden_vj, rel_emb):
        qst = torch.index_select(self.query_src_ts_emb, 0, torch.from_numpy(query_idx).long().to(self.device))
        qr = torch.index_select(self.query_rel_emb, 0, torch.from_numpy(query_idx).long().to(self.device))
        return self.transition_fn(((hidden_vi, rel_emb, qst, qr), (hidden_vj, rel_emb, qst, qr)))

    def forward(self, visited_node_score, selected_edges_l=None, visited_node_representation=None,
                rel_emb_l=None, max_edges=10, tc=None):
        transition_logits = self._cal_attention_score(selected_edges_l[-1], visited_node_representation, rel_emb_l[-1])
        src_score = visited_node_score[selected_edges_l[-1][:, -2]]
        transition_softmax = segment_softmax_op_v2(transition_logits, selected_edges_l[-1][:, -2], tc=tc)
        target_score = transition_softmax * src_score
        pruned_edges, pruned_target_score, orig_indices = self._topk_att_score(selected_edges_l[-1], target_score, max_edges)
        pruned_src_score = src_score[orig_indices]
        transition_pruned_softmax = transition_softmax[orig_indices]

        num_nodes = len(visited_node_representation)
        if self.node_score_aggregation == 'sum':
            sp_idx = torch.LongTensor(np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])).to(self.device)
            tm = torch.sparse.FloatTensor(sp_idx, transition_pruned_softmax, torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            updated_node_score = torch.squeeze(torch.sparse.mm(tm, pruned_src_score.unsqueeze(1)))
        else:
            raise ValueError("Only 'sum' aggregation supported in v2")

        updated_repr = self._update_node_repr(pruned_edges, visited_node_representation, transition_pruned_softmax, linear_act=False)
        for se, re in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            tl = self._cal_attention_score(se, updated_repr, re)
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
    def __init__(self, ngh_finder, rules_dict=None, num_entity=None, num_rel=None,
                 emb_dim: List[int] = None, DP_num_edges=40, DP_steps=3,
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

        self.temporal_embed_dim = [int(emb_dim[_] * 2 / (1 + emb_static_ratio)) for _ in range(DP_steps)]
        self.static_embed_dim = [emb_dim[_] * 2 - self.temporal_embed_dim[_] for _ in range(DP_steps)]

        self.entity_raw_embed = nn.Embedding(num_entity, self.static_embed_dim[0]).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = nn.Embedding(num_rel + 1, emb_dim[0]).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel

        self.att_flow_list = nn.ModuleList([
            AttentionFlow(emb_dim[_], emb_dim[_ + 1], node_score_aggregation=node_score_aggregation,
                          ratio_update=ratio_update, device=device)
            for _ in range(DP_steps)
        ])

        if use_time_embedding:
            self.node_emb_proj = nn.Linear(2 * emb_dim[0], emb_dim[0])
        else:
            self.node_emb_proj = nn.Linear(emb_dim[0], emb_dim[0])
        nn.init.xavier_normal_(self.node_emb_proj.weight)
        self.max_attended_edges = max_attended_edges

        if use_time_embedding:
            self.time_encoder = TimeEncode(expand_dim=self.temporal_embed_dim[0],
                                           entity_specific=diac_embed, num_entities=num_entity, device=device)
        self.ent_spec_time_embed = diac_embed
        self.device = device
        self.ent_score_aggregation = ent_score_aggregation

    def get_ent_emb(self, ent_idx_l, device):
        ed = next(self.entity_raw_embed.parameters()).get_device()
        ed = torch.device('cpu') if ed == -1 else torch.device(f'cuda:{ed}')
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l).long().to(ed)).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        ed = next(self.relation_raw_embed.parameters()).get_device()
        ed = torch.device('cpu') if ed == -1 else torch.device(f'cuda:{ed}')
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l).long().to(ed)).to(device)

    def get_node_emb(self, src_idx_l, cut_time_l, eg_idx):
        hidden_node = self.get_ent_emb(src_idx_l, self.device)
        if self.use_time_embedding:
            ct = cut_time_l - self.cut_time_l[eg_idx]
            if self.ent_spec_time_embed:
                ht = self.time_encoder(torch.from_numpy(ct[:, np.newaxis]).to(self.device), entities=src_idx_l)
            else:
                ht = self.time_encoder(torch.from_numpy(ct[:, np.newaxis]).to(self.device))
            return self.node_emb_proj(torch.cat([hidden_node, torch.squeeze(ht, 1)], axis=1))
        else:
            return self.node_emb_proj(hidden_node)

    def set_init(self, src_idx_l, rel_idx_l, cut_time_l):
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.cut_time_l = cut_time_l
        self.sampled_edges_l = []
        self.rel_emb_l = []
        self.node2index = {(i, src, ts): i for i, (src, rel, ts) in enumerate(zip(src_idx_l, rel_idx_l, cut_time_l))}
        self.num_existing_nodes = len(src_idx_l)

        # FIX 1: Build per-query rule mapping
        self.batch_rules = {}
        for i, rel in enumerate(rel_idx_l):
            rules = self.rules_dict.get(int(rel), [])
            if rules:
                self.batch_rules[i] = rules

        # FIX 1: Set per-query rules on neighbor finder (not union)
        self.ngh_finder.set_per_query_rules(self.batch_rules, current_hop=0)

        # Compute query embeddings
        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)
        if self.use_time_embedding:
            if self.ent_spec_time_embed:
                qts = self.time_encoder(torch.zeros(len(self.cut_time_l), 1).float().to(self.device), entities=self.src_idx_l)
            else:
                qts = self.time_encoder(torch.zeros(len(self.cut_time_l), 1).float().to(self.device))
            qts = torch.squeeze(qts, 1)
            query_src_ts_emb = self.node_emb_proj(torch.cat([query_src_emb, qts], axis=1))
        else:
            query_src_ts_emb = self.node_emb_proj(query_src_emb)

        for i, att_flow in enumerate(self.att_flow_list):
            if i > 0:
                query_src_ts_emb = self.att_flow_list[i - 1].bypass_forward(query_src_ts_emb)
                query_rel_emb = self.att_flow_list[i - 1].bypass_forward(query_rel_emb)
            att_flow.set_query_emb(query_src_ts_emb, query_rel_emb)

    def initialize(self):
        eg_idx_l = np.arange(len(self.src_idx_l), dtype=np.int32)
        att_score = np.ones_like(self.src_idx_l, dtype=np.float32) * (1 - 1e-8)
        attended_nodes = np.stack([eg_idx_l, self.src_idx_l, self.cut_time_l, np.arange(len(self.src_idx_l))], axis=1)
        visited_nodes_score = torch.from_numpy(att_score).to(self.device)
        return attended_nodes, attended_nodes, visited_nodes_score, self.att_flow_list[0].query_src_ts_emb

    def forward(self, sample):
        src_idx_l, rel_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.ts
        self.set_init(src_idx_l, rel_idx_l, cut_time_l)
        attended_nodes, visited_nodes, visited_node_score, visited_node_repr = self.initialize()

        for step in range(self.DP_steps):
            # FIX 1: Update per-query rules for current hop
            self.ngh_finder.set_per_query_rules(self.batch_rules, current_hop=step)

            attended_nodes, visited_nodes, visited_node_score, visited_node_repr = \
                self._flow(attended_nodes, visited_nodes, visited_node_score, visited_node_repr, step)
            visited_node_score = segment_norm_l1(visited_node_score, visited_nodes[:, 0])

        entity_att_score, entities = self.get_entity_attn_score(
            visited_node_score[attended_nodes[:, -1]], attended_nodes)
        return entity_att_score, entities

    def _flow(self, attended_nodes, visited_nodes, visited_node_score, visited_node_repr, step, tc=None):
        # FIX 3: Pass eg_idx to neighbor finder
        sampled_edges, new_sampled_nodes, new_attended_nodes = self._get_sampled_edges(
            attended_nodes, num_neighbors=self.DP_num_edges, step=step, add_self_loop=True, tc=tc)

        if len(new_sampled_nodes):
            new_emb = self.get_node_emb(new_sampled_nodes[:, 1], new_sampled_nodes[:, 2], eg_idx=new_sampled_nodes[:, 0])
            for i in range(step):
                new_emb = self.att_flow_list[i].bypass_forward(new_emb)
            visited_node_repr = torch.cat([visited_node_repr, new_emb], axis=0)
            visited_nodes = np.concatenate([visited_nodes, new_sampled_nodes], axis=0)

        self.sampled_edges_l.append(sampled_edges)
        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        for i in range(step):
            rel_emb = self.att_flow_list[i].bypass_forward(rel_emb)
        for j in range(step):
            self.rel_emb_l[j] = self.att_flow_list[step - 1].bypass_forward(self.rel_emb_l[j])
        self.rel_emb_l.append(rel_emb)

        new_score, updated_repr, pruned_edges, orig_indices = \
            self.att_flow_list[step](visited_node_score, selected_edges_l=self.sampled_edges_l,
                                     visited_node_representation=visited_node_repr,
                                     rel_emb_l=self.rel_emb_l, max_edges=self.max_attended_edges, tc=tc)

        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]
        _, indices = np.unique(pruned_edges[:, [0, 4, 3]], return_index=True, axis=0)
        updated_attended = pruned_edges[:, [0, 3, 4, 7]][indices]
        return updated_attended, visited_nodes, new_score, updated_repr

    def _get_sampled_edges(self, attended_nodes, num_neighbors=20, step=None, add_self_loop=True, tc=None):
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        node_idx_l = attended_nodes[:, 3]
        eg_idx_l = attended_nodes[:, 0]  # FIX 3: pass eg_idx for per-query matching

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = \
            self.ngh_finder.get_temporal_neighbor(src_idx_l, cut_time_l,
                                                   eg_idx_l=eg_idx_l,
                                                   num_neighbors=num_neighbors)

        if add_self_loop:
            src_ngh_node_batch = np.concatenate([src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1)
            src_ngh_eidx_batch = np.concatenate([src_ngh_eidx_batch, np.full((len(attended_nodes), 1), self.selfloop, dtype=np.int32)], axis=1)
            src_ngh_t_batch = np.concatenate([src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1)

        flat_node = src_ngh_node_batch.flatten()
        flat_eidx = src_ngh_eidx_batch.flatten()
        flat_t = src_ngh_t_batch.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + int(add_self_loop))
        mask = flat_node != -1

        sampled_edges = np.stack([
            eg_idx, np.repeat(src_idx_l, num_neighbors + int(add_self_loop)),
            np.repeat(cut_time_l, num_neighbors + int(add_self_loop)),
            flat_node, flat_t, flat_eidx,
            np.repeat(node_idx_l, num_neighbors + int(add_self_loop))
        ], axis=1)[mask]

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

        sampled_edges = np.concatenate([sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1)
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes) if new_sampled_nodes else np.array([]).reshape(0, 4)
        _, new_att_idx = np.unique(sampled_edges[:, [0, 4, 3]], return_index=True, axis=0)
        new_attended_nodes = sampled_edges[:, [0, 3, 4]][new_att_idx]
        return sampled_edges, new_sampled_nodes, new_attended_nodes

    def get_entity_attn_score(self, logits, nodes, tc=None):
        entity_attn_score, entities = self._aggregate_op_entity(logits, nodes, self.ent_score_aggregation)
        return entity_attn_score, entities

    def _aggregate_op_entity(self, logits, nodes, aggr='sum'):
        device = logits.get_device()
        device = torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}')
        num_nodes = len(nodes)
        entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
        sp_idx = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)]))
        sp_val = torch.ones(num_nodes, dtype=torch.float)
        if aggr == 'mean':
            c = Counter([(n[0], n[1]) for n in nodes[:, :2]])
            cnt = torch.tensor([c[(_[0], _[1])] for _ in nodes[:, :2]])
            sp_val = torch.div(sp_val, cnt)
        tm = torch.sparse.FloatTensor(sp_idx, sp_val, torch.Size([len(entities), num_nodes])).to(device)
        entity_att_score = torch.squeeze(torch.sparse.mm(tm, logits.unsqueeze(1)))
        return entity_att_score, entities

    def loss(self, entity_att_score, entities, target_idx_l, batch_size,
             gradient_iters_per_update=1, loss_fn='BCE'):
        one_hot = torch.from_numpy(
            np.array([int(v == target_idx_l[eg]) for eg, v in entities], dtype=np.float32)
        ).to(self.device)
        entity_att_score = entity_att_score * 0.999 + 0.0009
        if loss_fn == 'BCE':
            if gradient_iters_per_update == 1:
                return nn.BCELoss()(entity_att_score, one_hot)
            else:
                return nn.BCELoss(reduction='sum')(entity_att_score, one_hot) / (gradient_iters_per_update * batch_size)
        else:
            if gradient_iters_per_update == 1:
                return nn.NLLLoss()(entity_att_score, one_hot)
            else:
                return nn.NLLLoss(reduction='sum')(entity_att_score, one_hot) / (gradient_iters_per_update * batch_size)
