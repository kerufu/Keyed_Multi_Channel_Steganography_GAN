import layers

import numpy as np
import setting

he = layers.HammingCode()
hd = layers.HammingCode(decode_mode=True)

input_bits = np.random.choice(2, (setting.batch_size, setting.data_bit_size_per_channel))
print("input_bits", input_bits[0])
encoded_bits = he(input_bits)
print("encoded_bits[0]", encoded_bits[0])

for b in range(setting.batch_size):
    for index in range(setting.total_bit_size_per_channel):
        if np.random.rand() > 0.96:
            encoded_bits[b, index] = 1 - encoded_bits[b, index]
            

decoded_bits = hd(encoded_bits)
print("decoded_bits[0]", decoded_bits[0])


print(1-(sum(abs(input_bits[0]-decoded_bits[0]))/setting.data_bit_size_per_channel))