import os
os.chdir("..")
import collections

import cv2
import numpy as np
import tensorflow as tf

import dataset_worker
import GAN_worker
import setting
import character_mapper

dw = dataset_worker.dataset_worker()
ganw = GAN_worker.GAN_worker()
cm = character_mapper.character_mapper()

def master_process(image_path, commands):
    img = np.array([dw.preprocess_image(image_path)])
    img = img / 127.5 - 1

    commands_bits = []
    for index in range(setting.num_message_channel):
        cmd = commands[index]
        cmd = cm.int_to_bits(cmd)
        cmd = np.tile(cmd, setting.num_of_window_per_channel)
        commands_bits.append([cmd])

    output = (ganw.generator(img, commands_bits)[0] + 1) * 127.5

    cv2.imwrite(setting.sample_decoded_image, np.array(output))

def zombie_process(image_path, character_mapping=True):
    img = np.array([dw.preprocess_image(image_path)])
    img = img / 127.5 - 1

    commands = []

    for channel_index in range(setting.num_message_channel):
        cmd = ganw.decoders[channel_index](img)[0]
        cmd = tf.math.sigmoid(cmd)
        cmd = tf.math.round(cmd)
        cmd = np.array(cmd)
        cmd_voting = collections.defaultdict(int)
        for window_index in range(0, setting.num_of_window_per_channel, setting.coding_window_size):
            c = cm.bits_to_int(cmd[window_index:window_index+setting.coding_window_size])
            if character_mapping:
                c = cm.int_matching(c)
            cmd_voting[c] += 1
        commands.append(int(max(cmd_voting, key=cmd_voting.get)))

    return commands

def botnet_simulation():
    command_table = list(cm.mapping_table.keys())

    for vc in setting.vulnerable_command:
        command_table.remove(vc)
    print(command_table)

    evaluate_step = 1000000
    invalid_image_count = 0
    for _ in range(evaluate_step):
        commands = [command_table[int(np.random.randint(0, len(command_table)))] for _ in range(setting.num_message_channel)]
        master_process(setting.sample_image, commands)
        decoded_commands = zombie_process(setting.sample_decoded_image)
        for index in range(setting.num_message_channel):
            if commands[index] != decoded_commands[index]:
                print("command error: ", commands[index], decoded_commands[index])
                invalid_image_count += 1
                break
    print("Command accuracy: ", 1-invalid_image_count/evaluate_step)

# ganw.train(50000, dw.dataset)
# ganw.evaluate(dw.dataset, coding_mode=0)

botnet_simulation()
