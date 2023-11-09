import time

import tensorflow as tf
import numpy as np
from termcolor import cprint
import cv2

import GAN_definition
import setting
import layers
import worker_factory

class GAN_worker():
    def __init__(self, key, generator_iteration=1, discriminator_iteration=1, decoder_iteration=1, wgan=True) -> None:
        self.generator_iteration = generator_iteration
        self.discriminator_iteration = discriminator_iteration
        self.decoder_iteration = decoder_iteration

        self.generator = GAN_definition.generator(key)
        self.discriminator = GAN_definition.discriminator()
        self.decoders = [GAN_definition.decoder() for _ in range(setting.num_message_channel)]

        self.generator.load_weights(setting.GAN_pathes["generator"])
        self.discriminator.load_weights(setting.GAN_pathes["discriminator"])
        for index in range(setting.num_message_channel):
            self.decoders[index].load_weights(setting.GAN_pathes["decoder"]+str(index))
        print("GAN model weight loaded")

        try:
            self.generator.load_weights(setting.GAN_pathes["generator"])
            self.discriminator.load_weights(setting.GAN_pathes["discriminator"])
            for index in range(setting.num_message_channel):
                self.decoders[index].load_weights(setting.GAN_pathes["decoder"]+str(index))
            print("GAN model weight loaded")
        except:
            print("GAN model weight not found")

        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.decoder_opts = [tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay) for _ in range(setting.num_message_channel)]

        self.generator_loss = tf.keras.losses.MeanSquaredError()
        if wgan:
            self.discriminator_loss = layers.WassersteinLoss()
        else:
            self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)
        self.decoder_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_metric = tf.keras.metrics.MeanSquaredError()
        self.discriminator_metric = tf.keras.metrics.BinaryAccuracy(threshold=0) # for convenience, handle wgan score in the same way as logit, which is not actual
        self.decoders_metric = [tf.keras.metrics.BinaryAccuracy(threshold=0) for _ in range(setting.num_message_channel)]

    def get_generator_loss(self, input_image, output_image, messages, decoded_messages, discriminator_ouput_fake):
        loss = self.generator_loss(input_image, output_image) * setting.mse_weight
        decoders_loss = 0
        for index in range(setting.num_message_channel):
            decoders_loss += self.decoder_loss(messages[index], decoded_messages[index])
        loss += decoders_loss / setting.num_message_channel
        loss += self.discriminator_loss(tf.ones_like(discriminator_ouput_fake), discriminator_ouput_fake)
        loss += tf.add_n(self.generator.losses) * setting.regularization_weight
        return loss
    
    def get_discriminator_loss(self, target, output):
        return self.discriminator_loss(target, output) + tf.add_n(self.discriminator.losses) * setting.regularization_weight
    

    @tf.function
    def get_decoder_loss(self, message, decoded_message, index):
        return self.decoder_loss(message, decoded_message) + tf.add_n(self.decoders[index].losses) * setting.regularization_weight
    
    @tf.function
    def train_generator(self, batch):
        with tf.GradientTape() as generator_tape:
            messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            output_image = self.generator(batch, messages, training=True)

            decoded_messages = []
            for index in range(setting.num_message_channel):
                decoded_messages.append(self.decoders[index](output_image))
            
            discriminator_ouput_fake = self.discriminator(output_image)

            generator_loss = self.get_generator_loss(batch, output_image, messages, decoded_messages, discriminator_ouput_fake)
        
        generator_gradient = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_opt.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))

        self.generator_metric.update_state(batch, output_image)

    @tf.function
    def train_discriminator(self, batch):
        with tf.GradientTape() as discriminator_tape_true:
            discriminator_ouput_true = self.discriminator(batch, training=True)

            discriminator_loss_true = self.get_discriminator_loss(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)

        discriminator_gradient = discriminator_tape_true.gradient(discriminator_loss_true, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        with tf.GradientTape() as discriminator_tape_fake:
            messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            output_image = self.generator(batch, messages)
            discriminator_ouput_fake = self.discriminator(output_image, training=True)

            discriminator_loss_fake = self.get_discriminator_loss(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

        discriminator_gradient = discriminator_tape_fake.gradient(discriminator_loss_fake, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
        self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

    def train_decoder(self, batch, index):
        with tf.GradientTape() as decoder_tape:
            messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            output_image = self.generator(batch, messages)

            decoded_message = self.decoders[index](output_image, training=True)

            decoder_loss = self.get_decoder_loss(messages[index], decoded_message, index)

        decoder_gradient = decoder_tape.gradient(decoder_loss, self.decoders[index].trainable_variables)
        self.decoder_opts[index].apply_gradients(zip(decoder_gradient, self.decoders[index].trainable_variables))

        self.decoders_metric[index].update_state(messages[index], decoded_message)

    def train(self, epoch, dataset):
        dataset = dataset.take(setting.batch_size)
        dataset = dataset.shuffle(dataset.cardinality()//setting.shuffle_buffer_size_divider, reshuffle_each_iteration=True).batch(setting.batch_size, drop_remainder=True)

        acc_max = 0

        for epoch_num in range(epoch):
            start = time.time()
            self.generator_metric.reset_state()
            self.discriminator_metric.reset_state()
            for index in range(setting.num_message_channel):
                self.decoders_metric[index].reset_state()
            
            for batch in dataset:
                for _ in range(self.discriminator_iteration):
                    self.train_discriminator(batch)
                for _ in range(self.generator_iteration):
                    self.train_generator(batch)
                for _ in range(self.decoder_iteration):
                    for index in range(setting.num_message_channel):
                        self.train_decoder(batch, index)

            image = batch[:1, :]
            messages = [np.random.choice(2, (1, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            decoded_image = self.generator(image, messages)
            decoded_messages = [self.decoders[index](decoded_image) for index in range(setting.num_message_channel)]

            cv2.imwrite(setting.sample_image, np.array((image[0]+1)*127.5))
            cv2.imwrite(setting.sample_encoded_image, np.array((decoded_image[0]+1)*127.5))
            
            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Image Reconstruction Loss: " + str(self.generator_metric.result().numpy()))
            print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))

            acc_list = []
            for index in range(setting.num_message_channel):
                acc = self.decoders_metric[index].result().numpy()
                acc_list.append(acc)
                print("Message " + str(index+1) + " Reconstruction Accuracy: " + str(acc))
            acc_mean = np.mean(acc_list)
            if acc_mean > acc_max:
                acc_max = acc_mean
                if self.generator_iteration:
                    self.generator.save(setting.GAN_pathes["generator"])
                if self.discriminator_iteration:
                    self.discriminator.save(setting.GAN_pathes["discriminator"])
                if self.decoder_iteration:
                    for index in range(setting.num_message_channel):
                        self.decoders[index].save(setting.GAN_pathes["decoder"]+str(index))

            print("Message Reconstruction Average Accuracy: " + str(acc_mean))
            print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
            print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))

    def evaluate(self, dataset, coding_mode=0):
        dataset = dataset.take(setting.batch_size)
        dataset = dataset.shuffle(dataset.cardinality()).batch(setting.batch_size, drop_remainder=True)

        self.generator_metric.reset_state()
        self.discriminator_metric.reset_state()
        for index in range(setting.num_message_channel):
            self.decoders_metric[index].reset_state()

        enable_hamming = False
        enable_character_mapping= False
        if coding_mode == 1:
            enable_hamming = True
        elif coding_mode == 2:
            enable_character_mapping = True

        for batch in dataset:
            if enable_hamming:
                messages = [np.random.choice(2, (setting.batch_size, setting.data_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            elif enable_character_mapping:
                messages = []
                for _ in range(setting.num_message_channel):
                    m = np.random.choice(list(worker_factory.cm.mapping_table.keys()), (setting.batch_size, setting.num_of_window_per_channel))
                    m = np.expand_dims(m, axis=2)
                    m = np.apply_along_axis(lambda key: worker_factory.cm.mapping_table[int(key)], axis=2, arr=m)
                    m = tf.concat(m, axis=-1)
                    m = np.array(m)
                    m = m.reshape((-1, setting.total_bit_size_per_channel))
                    messages.append(m)
            else:
                messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]

            decoded_image = self.generator(batch, messages, enable_hamming=enable_hamming)
            decoded_messages = [self.decoders[index](decoded_image, enable_hamming=enable_hamming) for index in range(setting.num_message_channel)]
            
            if enable_character_mapping:
                for index in range(setting.num_message_channel):
                    decoded_messages[index] = tf.math.sigmoid(decoded_messages[index])
                    decoded_messages[index] = tf.math.round(decoded_messages[index])
                    decoded_messages[index] = np.array(decoded_messages[index])
                    decoded_messages[index] = decoded_messages[index].reshape((-1, setting.num_of_window_per_channel, setting.coding_window_size))
                    decoded_messages[index] = np.apply_along_axis(worker_factory.cm.bits_matching, axis=2, arr=decoded_messages[index])
                    decoded_messages[index] = decoded_messages[index].astype(np.float32)
                    decoded_messages[index] = decoded_messages[index].reshape((-1, setting.total_bit_size_per_channel))
                    decoded_messages[index] = decoded_messages[index] - 0.5

            discriminator_ouput_true = self.discriminator(batch)
            discriminator_ouput_fake = self.discriminator(decoded_image)
            
            self.generator_metric.update_state(batch, decoded_image)

            self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
            self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

            for index in range(setting.num_message_channel):
                self.decoders_metric[index].update_state(messages[index], decoded_messages[index])

            break

        cv2.imwrite(setting.sample_image, np.array((batch[0]+1)*127.5))
        cv2.imwrite(setting.sample_encoded_image, np.array((decoded_image[0]+1)*127.5))

        print("Image Reconstruction Loss: " + str(self.generator_metric.result().numpy()))
        print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))
        
        acc_list = []
        for index in range(setting.num_message_channel):
            acc = self.decoders_metric[index].result().numpy()
            acc_list.append(acc)
            print("Message " + str(index+1) + " Reconstruction Accuracy: " + str(acc))
        acc_mean = np.mean(acc_list)

        print("Message Reconstruction Average Accuracy: " + str(acc_mean))
        print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
        print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))



