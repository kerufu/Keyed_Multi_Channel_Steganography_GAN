import time

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dataset_path = "dataset/"
processed_dataset_path = "processed_dataset/"
shuffle_buffer_size_divider = 1

image_size = 64
batch_size = 100
num_batch = 1

GAN_pathes = {
    "generator": "saved_model/GAN/generator",
    "discriminator": "saved_model/GAN/discriminator",
    "decoder": "saved_model/GAN/decoder_",
    "authenticator": "saved_model/GAN/authenticator",
}

sample_image = "sample_image.png"
sample_encoded_image = "sample_encoded_image.png"
sample_downloaded_image = "sample_downloaded_image.png"
sample_reconstructed_image = "sample_reconstructed_image.png"
sample_compressed_image = "sample_compressed_image.jpg"

num_conv_channel = 32
kernal_size = 3

label_smoothing_ratio = 0.1

kernal_clip_value = 0.1

learning_rate = 0.0001
gradient_clip_norm = None
weight_decay = None

num_message_channel = 4

jpeg_compression_loss_weight = 0
jpeg_compression_iteration = 5

np.random.seed(0)
key_size = 32
GAN_key = np.random.choice(2, size=key_size)
np.random.seed(int(time.time()))

message_bit_per_pixel = 1
total_bit_size_per_channel = (image_size * image_size * message_bit_per_pixel) // num_message_channel
total_bit_size = total_bit_size_per_channel * num_message_channel

dropout_ratio = 0
regularization_weight = 0
discriminator_weight = 1
mse_weight_valid = 90
mse_weight_valid *= np.log2(message_bit_per_pixel*2) / message_bit_per_pixel
mse_weight_invalid = mse_weight_valid / 10
mse_weight_valid -= mse_weight_invalid
decoder_weight = 10
decoder_weight *= np.exp2(message_bit_per_pixel-1) / num_message_channel / message_bit_per_pixel
authenticator_weight = 1e-8 / message_bit_per_pixel

coding_window_size = 8

data_bit_size_per_window = np.log2(coding_window_size)
data_bit_size_per_window = int(coding_window_size-data_bit_size_per_window-1)
parity_bit_size_per_window = coding_window_size - data_bit_size_per_window - 1
num_of_window_per_channel = total_bit_size_per_channel // coding_window_size
residual_bits_size = total_bit_size_per_channel % coding_window_size
data_bit_size_per_channel = data_bit_size_per_window * num_of_window_per_channel + residual_bits_size
parity_indexes = []
for index in range(coding_window_size-1):
    log2_index = np.log2(index+1)
    if log2_index == int(log2_index):
        parity_indexes.append(index)

size_of_dictionary = 16  # if setting to 36, it can code a-z, 0-9. if setting to 16, then the utilization rate is the same as (8, 4) hamming code
code_space_size = 2 ** coding_window_size
mapping_table_path = "character_mapping_table.pickle"

command_set_path = "command_set.pickle"
vulnerable_command = {
    True: [

    ],
    False: [
    
    ]
}
command_set_seed = 7
command_repeat = 32
command_pad_width = total_bit_size_per_channel - command_repeat * coding_window_size

twitter_credential = {
    "bearer_key": "AAAAAAAAAAAAAAAAAAAAALSkqwEAAAAApvXQ6X3Um3R%2FLMmGCMpSmz%2BGxYc%3DoeVspNJF3SdPynckE8ORGhalGxz0bFiHUZESZsmcr5nZveaw1t",
    "api_key": "bxGtlYBwCHDs6wK48Yy6bbT8P",
    "api_secret": "qXj7IUhOH0JbbAM2wq2A68HfayA1f3rPoN8aDtkAWZ4I0SOQlQ",
    "access_token": "1721801499179974656-bewsgdBWw7t3DLXF3MrNCpUnsjJUGW",
    "access_token_secret": "a4R421fvuYUEpxouH9pZVNsIbXagHTTB0rpEAzgmvGUlj"
}