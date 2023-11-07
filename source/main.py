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

    output = cv2.cvtColor(np.array(output), cv2.COLOR_BGR2BGRA)
    output[:, :, 3] = 255
    output = cv2.copyMakeBorder(np.array(output), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value = 0)

    cv2.imwrite(setting.sample_encoded_image, output)

def zombie_process(image_path, character_mapping=False):

    img = np.array([cv2.imread(image_path, cv2.IMREAD_COLOR)])
    img = img[:, 1:-1, 1:-1, :]

    img = img / 127.5 - 1

    commands = []

    for channel_index in range(setting.num_message_channel):
        cmd = ganw.decoders[channel_index](img)[0]
        cmd = tf.math.sigmoid(cmd)
        cmd = tf.math.round(cmd)
        cmd = np.array(cmd)
        cmd_voting = collections.defaultdict(int)
        for window_index in range(0, setting.total_bit_size_per_channel, setting.coding_window_size):
            c = cm.bits_to_int(cmd[window_index:window_index+setting.coding_window_size])
            if character_mapping:
                c = cm.int_matching(c)
            cmd_voting[c] += 1
        commands.append(int(max(cmd_voting, key=cmd_voting.get)))

    return commands

def botnet_simulation(character_mapping=False):

    if character_mapping:
        command_table = list(cm.mapping_table.keys())
    else:
        command_table = list(range(2**setting.coding_window_size))

    for vc in setting.vulnerable_command[character_mapping]:
        command_table.remove(vc)

    simulate_step = 1000000
    vul_cmd = set([])
    for _ in range(simulate_step):
        commands = np.random.choice(command_table, setting.num_message_channel)
        master_process(setting.sample_image, commands)
        decoded_commands = zombie_process(setting.sample_encoded_image, character_mapping)
        print("commands: ", commands)
        print("decoded_commands: ", decoded_commands)
        for index in range(setting.num_message_channel):
            if commands[index] != decoded_commands[index]:
                print("command error: ", commands[index], decoded_commands[index])
                vul_cmd.add(commands[index])
                print(vul_cmd)

def generate_twitter_profile_image():
    command_table = list(range(2**setting.coding_window_size))
    for vc in setting.vulnerable_command[False]:
        command_table.remove(vc)
    commands = np.random.choice(command_table, setting.num_message_channel)
    print("commands: ", commands)
    master_process(setting.sample_image, commands)

def decode_twitter_profile_image(path):
    decoded_commands = zombie_process(path, False)
    print("decoded_commands: ", decoded_commands)

# ganw.train(50000, dw.dataset)
# ganw.evaluate(dw.dataset, coding_mode=0)

# botnet_simulation()

generate_twitter_profile_image()
# decode_twitter_profile_image("./Aq5wMfJw_400x400.png")

