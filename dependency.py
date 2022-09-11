import copy


class DepInstanceParser:
    def __init__(self, basicDependencies):
        self.basicDependencies = basicDependencies
        self.dep_governed_info, self.length = self.dep_parsing()

    def dep_parsing(self):
        dep_governed_info = {dep['dependent'] - 1: {
            "governor": dep['governor'] - 1,
            "dep": dep['dep']
            } for dep in self.basicDependencies}
        length = max([dep['dependent'] for dep in self.basicDependencies])
        return dep_governed_info, length

    def get_init_dep_matrix(self):
        dep_adj_matrix = [[0 for _ in range(self.length)] for _ in range(self.length)]
        dep_type_matrix = [["none" for _ in range(self.length)] for _ in range(self.length)]
        for i in range(self.length):
            dep_adj_matrix[i][i] = 1
            dep_type_matrix[i][i] = "self_loop"
        return dep_adj_matrix, dep_type_matrix

    def get_first_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_init_dep_matrix()

        for i, dep_info in self.dep_governed_info.items():
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)

        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][
                        second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix,
                                                          [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[start_index][right_index] = 1
                    dep_path_type_matrix[start_index][right_index] = first_order_dep_type_matrix[left_index][
                        right_index]
                    dep_path_adj_matrix[end_index][left_index] = 1
                    dep_path_type_matrix[end_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_global_graph(self, start_range, end_range, dep_type, direct=False):
        _, dep_path_type_matrix = self.get_init_dep_matrix()
        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        relation_position_e1 = ["none" for _ in first_order_dep_type_matrix]
        relation_position_e2 = ["none" for _ in first_order_dep_type_matrix]
        # Local
        for start_index in start_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] == "none":
                    continue
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
                if dep_type in ["D", "DS"]:
                    relation_position_e1[next_id] = first_order_dep_type_matrix[start_index][next_id]
        for start_index in end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] == "none":
                    continue
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
                if dep_type in ["D", "DS"]:
                    relation_position_e2[next_id] = first_order_dep_type_matrix[start_index][next_id]
        # Global
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix,
                                                          [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
                    if dep_type in ["S", "DS"]:
                        relation_position_e1[right_index] = "global"
                        relation_position_e2[left_index] = "global"
        # Inner
        for i in start_range:
            for j in start_range:
                if first_order_dep_type_matrix[i][j] != "none":
                    relation_position_e1[j] = "self_loop"
        for i in end_range:
            for j in end_range:
                if first_order_dep_type_matrix[i][j] != "none":
                    relation_position_e2[j] = "self_loop"
        return first_order_dep_type_matrix, dep_path_type_matrix, relation_position_e1, relation_position_e2

    def get_local_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range + end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] in ["none"]:
                    continue
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_global_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix,
                                                          [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix
