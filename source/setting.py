import math

image_size = 64

dataset_path = "dataset/"
processed_dataset_path = "processed_dataset/"

num_message = 2

feature_size = image_size * image_size
message_size = feature_size // 10 // num_message

keys = [
    [0] * (message_size - 1) + [0],
    [0] * (message_size - 1) + [1],
]

GAN_pathes = {
    "generator": "saved_model/GAN/generator",
    "discriminator": "saved_model/GAN/discriminator",
    "decoder": "saved_model/GAN/decoder",
}

batch_size = 50

dropout_ratio = 0.25

weight_decay = None

sample_image = "sample_image.jpg"
sample_decoded_image = "sample_decoded_image.jpg"

label_smoothing_ratio = 0.1
label_smoothing_logit_threshold = math.log(label_smoothing_ratio/(1-label_smoothing_ratio))

learning_rate = 0.0001
gradient_clip_norm = 1.0
regularization_weight = 0.1

kernal_clip_value = 0.05

shuffle_buffer_size_divider = 1

save_iteration = 1
