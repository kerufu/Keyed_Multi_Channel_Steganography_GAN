import os
os.chdir("..")

import cv2
import numpy as np

import dataset_worker
import GAN_worker
import setting

dw = dataset_worker.dataset_worker()
ganw = GAN_worker.GAN_worker()

def encrypt(image_path, messages):
    img = [dw.preprocess_image(image_path)]
    img = np.array(img)
    img = img / 127.5 - 1

    for index in range(setting.num_message_channel):
        if len(messages[index]) < setting.data_bit_size_per_window:
            messages[index] += [0] * (setting.data_bit_size_per_window - len(messages[index]))
        elif len(messages[index]) > setting.data_bit_size_per_window:
            messages[index] = messages[index][:setting.data_bit_size_per_window]

        messages[index] = np.array([messages[index]])

    output = ganw.generator(img, messages)

    cv2.imwrite(setting.sample_decoded_image, np.array((output[0]+1)*127.5))

def decrypt(image_path):
    img = [dw.preprocess_image(image_path)]
    img = np.array(img)
    img = img / 127.5 - 1

    for index in range(setting.num_message_channel):
        print(ganw.decoders[index](img))


# ganw.train(50000, dw.dataset)

ganw.evaluate(dw.dataset, coding_mode=0)

# encrypt(
#     "./dataset/101010.jpg",
#     [
#         [0, 1, 0, 0, 1, 1],
#         [1, 0, 1, 1, 0] * (setting.message_size // 3)
#     ]
# )

# decrypt(setting.sample_decoded_image)