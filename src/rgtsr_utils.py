"""
RGTSR Utilities - Unified data loading bridging TLogic's Grapher and xERTE's Data format.
Provides RGTSRData class that loads TKG data compatible with both rule learning and neural modules.
"""

import os
import json
import numpy as np
from collections import defaultdict, Counter


class RGTSRData:
    """
    Unified data loader for RGTSR model.
    Combines TLogic's Grapher (JSON-based entity/relation/ts mappings + inverse relations)
    with xERTE's Data format (integer-indexed txt files + adjacency structures).

    Supports both data formats:
      - TLogic format: entity2id.json, relation2id.json, ts2id.json, train.txt (tab-separated strings)
      - xERTE format: entity2id.txt, relation2id.txt, train.txt (tab-separated integers)

    Parameters:
        dataset_dir (str): path to the dataset directory
        add_reverse (bool): whether to add inverse relations (required for both TLogic and xERTE)
        data_format (str): 'tlogic' or 'xerte', determines how to parse files
    """

    def __init__(self, dataset_dir, add_reverse=True, data_format='tlogic'):
        self.dataset_dir = dataset_dir
        self.data_format = data_format
        self.add_reverse = add_reverse

        if data_format == 'tlogic':
            self._load_tlogic_format()
        elif data_format == 'xerte':
            self._load_xerte_format()
        else:
            raise ValueError("data_format must be 'tlogic' or 'xerte'")

        # Common post-processing
        self.num_entities = len(self.id2entity)
        self.num_relations_no_inv = self.num_relations_orig
        if add_reverse:
            self.num_relations = 2 * self.num_relations_orig
        else:
            self.num_relations = self.num_relations_orig

        # Add selfloop relation for xERTE module
        self.selfloop_rel = self.num_relations
        self.id2relation[self.selfloop_rel] = 'selfloop'

        # Build inverse relation mapping (needed by TLogic rule learning)
        self.inv_relation_id = {}
        for i in range(self.num_relations_orig):
            self.inv_relation_id[i] = i + self.num_relations_orig
        for i in range(self.num_relations_orig, 2 * self.num_relations_orig):
            self.inv_relation_id[i] = i % self.num_relations_orig

        # Add inverse quadruples to train/valid/test
        if add_reverse:
            self.train_idx = self._add_inverses(self.train_idx_orig)
            self.valid_idx = self._add_inverses(self.valid_idx_orig)
            self.test_idx = self._add_inverses(self.test_idx_orig)
        else:
            self.train_idx = self.train_idx_orig.copy()
            self.valid_idx = self.valid_idx_orig.copy()
            self.test_idx = self.test_idx_orig.copy()

        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        # Seen/unseen entity split for xERTE-style evaluation
        seen_entities = set(self.train_idx[:, 0]).union(set(self.train_idx[:, 2]))
        seen_relations = set(self.train_idx[:, 1])

        test_mask = np.array([
            (evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations)
            for evt in self.test_idx
        ])
        self.test_idx_seen = self.test_idx[test_mask]
        self.test_idx_unseen = self.test_idx[~test_mask]

        # Timestamps
        self.timestamps = np.array(sorted(set(self.all_idx[:, 3])))
        self.max_time = int(max(self.all_idx[:, 3]))

    def _load_tlogic_format(self):
        """Load data in TLogic format (JSON mappings, string-based txt files)."""
        self.entity2id = json.load(open(os.path.join(self.dataset_dir, "entity2id.json")))
        relation2id_orig = json.load(open(os.path.join(self.dataset_dir, "relation2id.json")))
        self.ts2id = json.load(open(os.path.join(self.dataset_dir, "ts2id.json")))

        self.relation2id = relation2id_orig.copy()
        self.num_relations_orig = len(relation2id_orig)

        # Add inverse relations
        counter = self.num_relations_orig
        for relation in relation2id_orig:
            self.relation2id["_" + relation] = counter
            counter += 1

        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.id2ts = {v: k for k, v in self.ts2id.items()}

        # Load and index quadruples
        self.train_idx_orig = self._load_tlogic_quads("train.txt")
        self.valid_idx_orig = self._load_tlogic_quads("valid.txt")
        self.test_idx_orig = self._load_tlogic_quads("test.txt")

    def _load_tlogic_quads(self, filename):
        """Load quadruples from TLogic-format file (string-based, tab-separated)."""
        filepath = os.path.join(self.dataset_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        quads = [line.strip().split("\t") for line in lines if line.strip()]
        subs = [self.entity2id[x[0]] for x in quads]
        rels = [self.relation2id[x[1]] for x in quads]
        objs = [self.entity2id[x[2]] for x in quads]
        tss = [self.ts2id[x[3]] for x in quads]
        return np.column_stack((subs, rels, objs, tss))

    def _load_xerte_format(self):
        """Load data in xERTE format (integer-indexed txt files)."""
        # Entity mapping
        self.id2entity = {}
        with open(os.path.join(self.dataset_dir, "entity2id.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                self.id2entity[int(parts[1].strip())] = parts[0].strip()
        self.entity2id = {v: k for k, v in self.id2entity.items()}

        # Relation mapping
        self.id2relation = {}
        with open(os.path.join(self.dataset_dir, "relation2id.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                self.id2relation[int(parts[1].strip())] = parts[0].strip()
        self.relation2id = {v: k for k, v in self.id2relation.items()}
        self.num_relations_orig = len(self.id2relation)

        # Add inverse relations to mappings
        for i in range(self.num_relations_orig):
            inv_name = 'Reversed ' + self.id2relation[i]
            inv_id = i + self.num_relations_orig
            self.id2relation[inv_id] = inv_name
            self.relation2id[inv_name] = inv_id

        # Timestamp mapping (auto-generated for xERTE format)
        self.ts2id = {}
        self.id2ts = {}

        # Load integer quadruples
        self.train_idx_orig = self._load_xerte_quads("train.txt")
        self.valid_idx_orig = self._load_xerte_quads("valid.txt")
        self.test_idx_orig = self._load_xerte_quads("test.txt")

        # Build ts2id from all data
        all_ts = set(self.train_idx_orig[:, 3]) | set(self.valid_idx_orig[:, 3]) | set(self.test_idx_orig[:, 3])
        for ts in sorted(all_ts):
            self.ts2id[str(ts)] = int(ts)
            self.id2ts[int(ts)] = str(ts)

    def _load_xerte_quads(self, filename):
        """Load quadruples from xERTE-format file (integer-based, tab-separated)."""
        filepath = os.path.join(self.dataset_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data = np.array([[int(x.strip()) for x in line.split("\t")] for line in lines if line.strip()])
        return data[:, :4]  # Ensure only (s, r, o, t)

    def _add_inverses(self, quads_idx):
        """Add inverse quadruples: (o, r_inv, s, t) for each (s, r, o, t)."""
        subs = quads_idx[:, 2]
        rels = np.array([self.inv_relation_id[r] for r in quads_idx[:, 1]])
        objs = quads_idx[:, 0]
        tss = quads_idx[:, 3]
        inv_quads = np.column_stack((subs, rels, objs, tss))
        return np.vstack((quads_idx, inv_quads))

    # ---- Adjacency structures for xERTE's NeighborFinder ----

    def get_adj_dict(self):
        """Get adjacency dict: entity -> list of (object, relation, timestamp), sorted by (t, o, r)."""
        adj_dict = defaultdict(list)
        for event in self.all_idx:
            adj_dict[int(event[0])].append((int(event[2]), int(event[1]), int(event[3])))
        for v in adj_dict.values():
            v.sort(key=lambda x: (x[2], x[0], x[1]))
        return adj_dict

    def get_sp2o(self):
        """Mapping (subject, predicate) -> list(objects) over ALL data."""
        sp2o = defaultdict(list)
        for event in self.all_idx:
            sp2o[(event[0], event[1])].append(event[2])
        return sp2o

    def get_spt2o(self, dataset='valid'):
        """Mapping (subject, predicate, timestamp) -> list(objects)."""
        if dataset == 'train':
            events = self.train_idx
        elif dataset == 'valid':
            events = self.valid_idx
        elif dataset == 'test':
            events = self.test_idx
        else:
            raise ValueError("dataset must be 'train', 'valid', or 'test'")
        spt2o = defaultdict(list)
        for event in events:
            spt2o[(event[0], event[1], event[3])].append(event[2])
        return spt2o

    # ---- Edge structures for TLogic's rule learning ----

    def get_edges_by_relation(self, data=None):
        """Store edges grouped by relation (for TLogic rule learning/application)."""
        if data is None:
            data = self.train_idx
        edges = {}
        relations = list(set(data[:, 1]))
        for rel in relations:
            edges[rel] = data[data[:, 1] == rel]
        return edges

    def get_neighbors_by_node(self, data=None):
        """Store outgoing edges grouped by node (for TLogic temporal walks)."""
        if data is None:
            data = self.train_idx
        neighbors = {}
        nodes = list(set(data[:, 0]))
        for node in nodes:
            neighbors[node] = data[data[:, 0] == node]
        return neighbors


def store_edges(quads):
    """Store edges grouped by relation (standalone function for compatibility with TLogic)."""
    edges = {}
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]
    return edges


def calculate_obj_distribution(learn_data, edges):
    """Calculate object distribution for baseline fallback."""
    objects = learn_data[:, 2]
    dist = Counter(objects)
    for obj in dist:
        dist[obj] /= len(learn_data)
    obj_dist = {k: round(v, 6) for k, v in dist.items()}
    obj_dist = dict(sorted(obj_dist.items(), key=lambda item: item[1], reverse=True))

    rel_obj_dist = {}
    for rel in edges:
        objects = edges[rel][:, 2]
        dist = Counter(objects)
        for obj in dist:
            dist[obj] /= len(objects)
        rel_obj_dist[rel] = {k: round(v, 6) for k, v in dist.items()}
        rel_obj_dist[rel] = dict(sorted(rel_obj_dist[rel].items(), key=lambda item: item[1], reverse=True))

    return obj_dist, rel_obj_dist
