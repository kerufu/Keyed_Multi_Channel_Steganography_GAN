import os
os.chdir("..")

import layers

import numpy as np
import setting
import worker_pool
import math
import cv2
import tensorflow as tf


acc = 0.965

def hamming_test():

    he = layers.HammingCode()
    hd = layers.HammingCode(decode_mode=True)

    input_bits = np.random.choice(2, (setting.batch_size, setting.data_bit_size_per_channel))
    encoded_bits = he(input_bits)

    for b in range(setting.batch_size):
        for index in range(setting.total_bit_size_per_channel):
            if np.random.uniform() > acc:
                encoded_bits[b, index] = 1 - encoded_bits[b, index]
                
    decoded_bits = hd(encoded_bits)

    new_acc = 1-(sum(abs(input_bits[0]-decoded_bits[0]))/setting.data_bit_size_per_channel)
    print("final acc", new_acc)

def character_mapper_evaluate():

    cm = worker_pool.cm
    print(cm.mapping_table.keys())

    dis_sum = 0
    evaluate_size = 512
    for _ in range(evaluate_size):
        word = np.random.choice(list(cm.mapping_table.keys()))
        word_original = word
        for bit_index in range(setting.coding_window_size):
            if np.random.uniform() > acc:
                word = np.bitwise_xor(word, 2**bit_index)
        mapped_word = cm.bits_matching(cm.int_to_bits(word))
        mapped_word = cm.bits_to_int(mapped_word)
        dis = cm.hamming_distance(word_original, mapped_word)
        dis_sum += dis
    print("final acc", 1-(dis_sum/evaluate_size/setting.coding_window_size))

def botnet_metrics(num_redundacy):
    def prob(total, num, p):
        choice = math.comb(total, num)
        return choice * (p ** (num)) * ((1-p) ** (total-num))
    
    byte_acc = acc ** 8
    byte_err_rate = (1 - byte_acc) / 255
    result = 0

    for i in range(num_redundacy//2):
        result += prob(num_redundacy, num_redundacy-i, byte_acc)

    for i in range(num_redundacy//2, num_redundacy+1):
        prob_corr = prob(num_redundacy, num_redundacy-i, byte_acc)
        for j in range(num_redundacy-i):
            prob_err = prob(i, j, byte_err_rate)
            result += prob_corr * prob_err
        prob_err = prob(i, num_redundacy-i, byte_err_rate)
        result += prob_corr * prob_err / 2

    print("num bot: ", setting.total_bit_size//8//num_redundacy)
    print("acc: ", result)

def jpeg_compression_testing(image_path):
    import dataset_worker
    dw = dataset_worker.dataset_worker()
    img = dw.preprocess_image(image_path)
    while True:
        # cv2.imwrite("sample_compressed_image.jpg", (img+1)*127.5)
        # new_img = dw.preprocess_image("sample_compressed_image.jpg")
        result, ci = cv2.imencode('.jpg', (img+1)*127.5)
        new_img = cv2.imdecode((ci), 1) / 127.5 - 1
        print(np.mean(np.abs(new_img-img)))
        img = new_img

        cv2.imwrite(setting.sample_compressed_image, (img+1)*127.5)
        cv2.waitKey(100)
        
def xor_test():
    model = tf.keras.Sequential(
    [
        layers.EncrypteMessage(setting.GAN_key),
        tf.keras.layers.Dense(setting.total_bit_size_per_channel, activation="tanh"),
        tf.keras.layers.Dense(setting.total_bit_size_per_channel)
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=tf.keras.metrics.BinaryAccuracy(threshold=0)
    )
    for _ in range(100):
        messages = np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel))
        model.fit(
            x=messages,
            y=messages,
            batch_size=setting.batch_size
        )

# character_mapper_evaluate()


