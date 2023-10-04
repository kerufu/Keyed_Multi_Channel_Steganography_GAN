import math

image_size = 64

dataset_path = "dataset/"
processed_dataset_path = "processed_dataset/"

num_message_channel = 4

message_bit_per_pixel = 1
totoal_message_size = int(image_size*image_size*message_bit_per_pixel)
message_size = totoal_message_size // num_message_channel

keys = [
    [0] * message_size,
    [0, 1] * (message_size // 2),
]

GAN_pathes = {
    "generator": "saved_model/GAN/generator",
    "discriminator": "saved_model/GAN/discriminator",
    "decoder": "saved_model/GAN/decoder",
}

batch_size = 50

dropout_ratio = 0.25


sample_image = "sample_image.png"
sample_decoded_image = "sample_decoded_image.png"

label_smoothing_ratio = 0.1

learning_rate = 0.0001
gradient_clip_norm = None
weight_decay = None

regularization_weight = 0
mse_weight = 100

kernal_clip_value = 0.1

shuffle_buffer_size_divider = 1

