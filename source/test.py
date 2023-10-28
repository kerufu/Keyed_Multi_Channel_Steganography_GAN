import layers

import numpy as np
import setting
import character_mapper

def hamming_test():

    acc = 0.96

    he = layers.HammingCode()
    hd = layers.HammingCode(decode_mode=True)

    input_bits = np.random.choice(2, (setting.batch_size, setting.data_bit_size_per_channel))
    print("input_bits", input_bits[0])
    encoded_bits = he(input_bits)
    print("encoded_bits[0]", encoded_bits[0])

    for b in range(setting.batch_size):
        for index in range(setting.total_bit_size_per_channel):
            if np.random.rand() > acc:
                encoded_bits[b, index] = 1 - encoded_bits[b, index]
                break
                

    decoded_bits = hd(encoded_bits)
    print("decoded_bits[0]", decoded_bits[0])

    new_acc = 1-(sum(abs(input_bits[0]-decoded_bits[0]))/setting.data_bit_size_per_channel)
    print("acc improvement", (new_acc-acc)/acc)

def character_mapper_evaluate():

    cm = character_mapper.character_mapper()

    print(cm.mapping_table.keys())
    print(cm.mapping_table.values())

    input_bits = np.random.choice(2, setting.coding_window_size)
    output_bits = cm.matching([0, 0, 0, 0, 0, 0, 0, 0])
    print(input_bits)
    print(output_bits)

    dis_sum = 0
    evaluate_size = 1000000
    count = 0
    while count < evaluate_size:
        word = np.random.choice(list(cm.mapping_table.keys()))
        word_original = word
        for bit_index in range(setting.coding_window_size):
            if np.random.uniform() > 0.965:
                word = np.bitwise_xor(word, 2**bit_index)
        # print(self.int_to_bits(word_original), self.int_to_bits(word))
        mapped_word = cm.matching(cm.int_to_bits(word))
        mapped_word = cm.bits_to_int(mapped_word)
        dis = cm.hamming_distance(word_original, mapped_word)
        dis_sum += dis
        count += 1
    print(1-dis_sum/evaluate_size/setting.coding_window_size)