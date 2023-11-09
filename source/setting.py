import numpy as np

image_size = 64

dataset_path = "dataset/"
processed_dataset_path = "processed_dataset/"

num_message_channel = 4

message_bit_per_pixel = 1
total_bit_size = int(image_size*image_size*message_bit_per_pixel)
total_bit_size_per_channel = total_bit_size // num_message_channel

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
    True: [247, 8, 62],
    False: [
        0, 1, 128, 2, 4, 3, 6, 129, 8, 12, 16, 24,
        159, 32, 175, 48, 191, 64, 63, 192, 207, 223,
        231, 239, 243, 252, 247, 249, 126, 251, 127,
        253, 254, 255, 96, 235, 95, 199, 245, 250, 62, 56, 227
        ]
    }
command_set_seed = 7

GAN_pathes = {
    "generator": "saved_model/GAN/generator",
    "discriminator": "saved_model/GAN/discriminator",
    "decoder": "saved_model/GAN/decoder",
}

AE_pathes = {
    "encoder": "saved_model/AE/encoder",
    "decoder": "saved_model/AE/decoder",
}

batch_size = 50

dropout_ratio = 0.25

sample_image = "sample_image.png"
sample_encoded_image = "sample_encoded_image.png"
sample_downloaded_image = "sample_downloaded_image.png"
sample_reconstructed_image = "sample_reconstructed_image.png"

label_smoothing_ratio = 0.1

learning_rate = 0.0001
gradient_clip_norm = None
weight_decay = None

regularization_weight = 0
mse_weight = 100

kernal_clip_value = 0.1

shuffle_buffer_size_divider = 1

np.random.seed(0)

GAN_key = [0] * total_bit_size_per_channel
# GAN_key = np.random.choice(2, size=total_bit_size_per_channel)

AE_feature_size = 128
AE_key = np.random.choice(2, size=(image_size, image_size, AE_feature_size))

twitter_credential = {
    "bearer_key": "AAAAAAAAAAAAAAAAAAAAALSkqwEAAAAALv5wu%2BTFlhkGnHJxyWQvJo3Opcc%3DiFJ85iK6k2MM02Ien9prOfgBdo6rdov8kgzgh6NBjuQZN9jRKK",
    "api_key": "tMEQCcXEmTySJkLILF5IycchS",
    "api_secret": "xO0bHt6YfpAfvH8ETrBNbQMBWdyEQrEA5wp9xokc34pynxroCg",
    "access_token": "1721801499179974656-YAVkRZCeH2wDR6uTeAzOJXpWoZ9wjt",
    "access_token_secret": "ttOPqSxHazyoYWSHWXdnyrLl6siQpcntgstEWbYSk1cZ1"
}

