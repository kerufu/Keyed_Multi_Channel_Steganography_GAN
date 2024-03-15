import numpy as np
import pickle

import setting


class character_mapper():
    def __init__(self) -> None:

        try:
            with open(setting.mapping_table_path, 'rb') as fp:
                self.mapping_table = pickle.load(fp)
        except Exception as e:
            print(e)
            self.compute_mapping_table()

    def hamming_distance(self, n1, n2):
        return bin(np.bitwise_xor(np.uint64(n1), np.uint64(n2))).count("1")

    def distance_mean_std(self, index):
        distance = []
        for ci in self.mapping_table:
            distance.append(self.hamming_table[ci][index])
        return np.mean(distance), np.std(distance)

    def selective_generation(self):
        self.mapping_table = [setting.command_set_seed]
        while len(self.mapping_table) < setting.size_of_dictionary:
            max_mean = -np.inf
            min_std = np.inf
            new_code_index = 0
            for index in range(setting.code_space_size):
                if not index in self.mapping_table:
                    mean, std = self.distance_mean_std(index)
                    if (mean > max_mean) or (mean == max_mean and std < min_std):
                        max_mean = mean
                        min_std = std
                        new_code_index = index
            self.mapping_table.append(new_code_index)

    def reductive_generation(self):
        self.mapping_table = [i for i in range(setting.code_space_size)]
        while len(self.mapping_table) > setting.size_of_dictionary:
            min_dis_sum = np.inf
            min_index = 0
            np.random.shuffle(self.mapping_table)
            for i in self.mapping_table:
                dis_sum = np.sum(self.hamming_table[i, :])
                if dis_sum < min_dis_sum:
                    min_dis_sum = dis_sum
                    min_index = i
            self.hamming_table[min_index, :] = 0
            self.hamming_table[:, min_index] = 0
            self.mapping_table.remove(min_index)

    def compute_mapping_table(self, selective=True):
        self.hamming_table = np.zeros(
            (setting.code_space_size, setting.code_space_size), dtype=int)
        for i in range(setting.code_space_size):
            for j in range(i+1, setting.code_space_size):
                self.hamming_table[i, j] = self.hamming_distance(i, j)
        self.hamming_table += self.hamming_table.T

        if selective:
            self.selective_generation()
        else:
            self.reductive_generation()

        avg_hamming_dist = 0
        hamming_dist_count = 0
        for i in range(setting.size_of_dictionary):
            for j in range(i+1, setting.size_of_dictionary):
                hamming_dist_count += 1
                avg_hamming_dist += self.hamming_distance(
                    self.mapping_table[i], self.mapping_table[j])
        print(" Mapping Table Average Hamming Distance:",
              avg_hamming_dist/hamming_dist_count)

        mapping_table = {}
        for code_index in range(setting.size_of_dictionary):
            bits = self.int_to_bits(self.mapping_table[code_index])
            mapping_table[self.mapping_table[code_index]] = bits
        self.mapping_table = mapping_table

        with open(setting.mapping_table_path, 'wb') as fp:
            pickle.dump(self.mapping_table, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def bits_to_int(self, bits):
        bits_int = 0
        for i, j in enumerate(bits[::-1]):
            bits_int += np.left_shift(np.uint64(j), np.uint64(i))
        return bits_int

    def int_to_bits(self, integer):
        bits = np.binary_repr(integer, width=setting.coding_window_size)
        bits = np.fromstring(bits, 'u1') - ord('0')
        return bits

    def bits_matching(self, bits):
        bits_int = self.bits_to_int(bits)
        min_dis = np.inf
        min_code = 0
        for code_int in self.mapping_table.keys():
            dis = self.hamming_distance(code_int, bits_int)
            if dis < min_dis:
                min_dis = dis
                min_code = code_int
        return self.mapping_table[min_code]

    def int_matching(self, ints):
        min_dis = np.inf
        min_code = 0
        for code_int in self.mapping_table.keys():
            dis = self.hamming_distance(code_int, ints)
            if dis < min_dis:
                min_dis = dis
                min_code = code_int
        return self.bits_to_int(self.mapping_table[min_code])
