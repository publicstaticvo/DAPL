# coding:utf-8

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from dependency import DepInstanceParser

"""
Data format:
<instance id>\001<relation id>\001<start position of the first entity>\001
<end position of the first entity>\001<start position of the second entity>\001
<end position of the second entity>\001<sentence split by \002>\001
<valid flag split by \002>\001<dependency split by \002 and \003>(optional)
"""


class Data(Dataset):

    def __init__(self, data_path, max_length, n=5, k=5, steps=None, dep_file=None,
                 dep_type="DS", direct=False, model="none"):
        """
        inst_id_detail:dict example index -> (word_ids, e1_pos, e2_pos, relation_id)
            word_ids:list of word id in a sentence
            e1_pos, e2_pos:int pos of entities
            relation_id:int id of relation
        relation_name_id:dict relation name -> relation id
        relation_inst_ids:dict relation_id -> example ids with the targeted relation
        """
        self.max_length = max_length
        self.direct = direct
        self.steps = steps
        self.model = model
        self.n = n
        self.k = k
        self.types_dict = self.load_type_dict(dep_file)
        self.id2label, self.relation_inst_ids, self.inst_id_detail = self.load_data(data_path, dep_type)

    def load_type_dict(self, dep_file):
        dep_labels = ["self_loop", "global"]
        with open(dep_file, 'r') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if self.direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        types_dict = {"none": 0}
        for dep_type in dep_labels:
            types_dict[dep_type] = len(types_dict)
        return types_dict

    def load_data(self, data_path, dep_type):
        L = self.max_length
        relation_name_id = {}
        relation_inst_ids = {}
        inst_id_detail = {}
        relations = []
        for line in open(data_path):
            idx, label, words, valid, e11m, e12m, e21m, e22m, dep = line.strip().split('\001')
            e11m, e12m, e21m, e22m = map(lambda x: int(x), [e11m, e12m, e21m, e22m])
            # input_id
            words = [int(x) for x in words.split('\002')]
            # valid
            valid_flag = []
            valid = [int(x) for x in valid.split('\002')] + [len(words) - 1]
            words_count = len(valid) - 1
            for i in range(len(valid) - 1):
                valid_flag.append([0 for _ in range(valid[i])] + [1 for _ in range(valid[i], valid[i + 1])]
                                  + [0 for _ in range(L - valid[i + 1])])
            valid_flag.extend([[0 for _ in range(L - 1)] + [1] for _ in range(L - len(valid_flag))])
            # label
            if label not in relation_name_id:
                relation_name_id[label] = len(relation_name_id.keys())
                relations.append(relations)
            label = relation_name_id[label]
            # attention_mask
            mask = [1 for _ in words] + [0 for _ in range(L - len(words))]
            words += [0 for _ in range(L - len(words))]
            # e_mask
            if self.model in ["gcn", "gat", "agcn"]:
                e11, e12, e21, e22 = e11m, e12m, e21m, e22m
            else:
                e11 = words.index(1)
                e12 = words.index(2)
                e21 = words.index(3)
                e22 = words.index(4)
            e1_mask = [0 for _ in range(e11)] + [1 for _ in range(e11, e12 + 1)] + [0 for _ in range(e12 + 1, L)]
            e2_mask = [0 for _ in range(e21)] + [1 for _ in range(e21, e22 + 1)] + [0 for _ in range(e22 + 1, L)]
            # dependency
            dep_info = []
            for items in dep.split('\002'):
                items = items.split('\003')
                dep_info.append({"governor": int(items[0]), "dependent": int(items[1]), "dep": items[2]})
            dep_instance_parser = DepInstanceParser(dep_info)
            full_graph, lg_graph, dp1, dp2 = dep_instance_parser.get_local_global_graph(
                list(range(e11m, e12m + 1)), list(range(e21m, e22m + 1)), dep_type, direct=self.direct)
            dep_type_matrix = np.zeros((L, L), dtype=np.int).tolist()
            for pi in range(words_count):
                for pj in range(words_count):
                    dep_type_matrix[pi][pj] = self.types_dict[lg_graph[pi][pj]]
            # cast from valid to all
            fdp = np.zeros((L, 2), dtype=np.int)
            for i in range(len(valid) - 1):
                fdp[valid[i]:valid[i + 1], 0] = self.types_dict[dp1[i]]
                fdp[valid[i]:valid[i + 1], 1] = self.types_dict[dp2[i]]
            detail = [words, mask, e1_mask, e2_mask, valid_flag, dep_type_matrix, fdp.tolist()]
            inst_id_detail[idx] = detail
            inst_list = relation_inst_ids.get(label, [])
            inst_list.append(idx)
            relation_inst_ids[label] = inst_list
        return relations, relation_inst_ids, inst_id_detail

    def sample(self, item_list, count):
        if len(item_list) < count:
            idx = item_list[0]
            idxs = [idx] * count
        else:
            idxs = random.sample(item_list, count)
        return idxs

    def collect(self, idxs):
        input_ids, mask, e1_mask, e2_mask, valid, dep_type, dp = [], [], [], [], [], [], []
        for idx in idxs:
            input_ids.append(self.inst_id_detail[idx][0])
            mask.append(self.inst_id_detail[idx][1])
            e1_mask.append(self.inst_id_detail[idx][2])
            e2_mask.append(self.inst_id_detail[idx][3])
            valid.append(self.inst_id_detail[idx][4])
            dep_type.append(self.inst_id_detail[idx][5])
            dp.append(self.inst_id_detail[idx][6])
        return input_ids, mask, e1_mask, e2_mask, valid, dep_type, dp

    def get_mask(self, id_list):
        mask = []
        for l in id_list:
            mask.append([1] * len(l))
        return mask

    def padding(self, item_list, default):
        new_out = []
        for l in item_list:
            l = l[:self.max_length]
            l.extend([default] * (self.max_length - len(l)))
            new_out.append(l)
        return new_out

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        relations = random.sample(self.relation_inst_ids.keys(), self.n)
        support_idx = []
        query_idx = []
        for r in relations:
            idx = self.sample(self.relation_inst_ids[r], self.k + 1)
            support_idx.extend(idx[:-1])
            query_idx.append(idx[-1])
        s_input_ids, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp = self.collect(support_idx)
        q_input_ids, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp = self.collect(query_idx)
        s_input_ids, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp, q_input_ids, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp, relations = map(
            lambda x: torch.tensor(x),
            [s_input_ids, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp, q_input_ids, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp, relations]
        )
        return s_input_ids, s_mask, s_e1_mask, s_e2_mask, s_valid, s_dep_type, s_dp, q_input_ids, q_mask, q_e1_mask, q_e2_mask, q_valid, q_dep_type, q_dp, relations
