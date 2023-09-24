import time

import tensorflow as tf
import numpy as np
from termcolor import cprint
import cv2

import GAN_definition
import setting

class GAN_worker():
    def __init__(self, generator_iteration=1, discriminator_iteration=1, decoder_iteration=1) -> None:
        self.generator_iteration = generator_iteration
        self.discriminator_iteration = discriminator_iteration
        self.decoder_iteration = decoder_iteration

        self.generator = GAN_definition.generator(setting.keys[1])
        self.discriminator = GAN_definition.discriminator()
        self.decoders = [GAN_definition.decoder() for _ in range(setting.num_message)]

        try:
            self.generator.load_weights(setting.GAN_pathes["generator"])
            self.discriminator.load_weights(setting.GAN_pathes["discriminator"])
            for index in range(setting.num_message):
                self.decoders[index].load_weights(setting.GAN_pathes["decoder"]+str(index))
            print("GAN model weight loaded")
        except:
            print("GAN model weight not found")

        # self.generator_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        # self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        # self.decoder_opt = tf.keras.optimizers.legacy.Adam(learning_rate=setting.learning_rate)

        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate)
        self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=setting.learning_rate)
        self.decoder_opt = tf.keras.optimizers.legacy.Adam(learning_rate=setting.learning_rate)


        self.generator_loss = tf.keras.losses.MeanSquaredError()
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)
        self.decoder_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_metric = tf.keras.metrics.MeanSquaredError()
        self.discriminator_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.decoders_metric = [tf.keras.metrics.BinaryAccuracy(threshold=0) for _ in range(setting.num_message)]

    def get_generator_loss(self, input_image, output_image, messages, decoded_messages, discriminator_ouput_fake):
        loss = self.generator_loss(input_image, output_image)
        decode_loss = 0
        for index in range(setting.num_message):
            decode_loss  += self.decoder_loss(messages[index], decoded_messages[index])
        loss += decode_loss / setting.num_message
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
            messages = [np.random.choice(2, (setting.batch_size, setting.message_size)) for _ in range(setting.num_message)]
            output_image = self.generator(batch, messages, training=True)

            decoded_messages = []
            for index in range(setting.num_message):
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
            messages = [np.random.choice(2, (setting.batch_size, setting.message_size)) for _ in range(setting.num_message)]
            output_image = self.generator(batch, messages)
            discriminator_ouput_fake = self.discriminator(output_image, training=True)

            discriminator_loss_fake = self.get_discriminator_loss(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

        discriminator_gradient = discriminator_tape_fake.gradient(discriminator_loss_fake, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
        self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

    def train_decoder(self, batch, index):
        with tf.GradientTape() as decoder_tape:
            messages = [np.random.choice(2, (setting.batch_size, setting.message_size)) for _ in range(setting.num_message)]
            output_image = self.generator(batch, messages)

            decoded_message = self.decoders[index](output_image, training=True)

            decoder_loss = self.get_decoder_loss(messages[index], decoded_message, index)

        decoder_gradient = decoder_tape.gradient(decoder_loss, self.decoders[index].trainable_variables)
        self.decoder_opt.apply_gradients(zip(decoder_gradient, self.decoders[index].trainable_variables))

        self.decoders_metric[index].update_state(messages[index], decoded_message)

    def train(self, epoch, dataset):
        dataset = dataset.take(setting.batch_size)
        dataset = dataset.shuffle(dataset.cardinality()//setting.shuffle_buffer_size_divider, reshuffle_each_iteration=True).batch(setting.batch_size, drop_remainder=True)

        for epoch_num in range(epoch):
            start = time.time()
            self.generator_metric.reset_state()
            self.discriminator_metric.reset_state()
            for index in range(setting.num_message):
                self.decoders_metric[index].reset_state()
            
            for batch in dataset:
                for _ in range(self.discriminator_iteration):
                    self.train_discriminator(batch)
                for _ in range(self.generator_iteration):
                    self.train_generator(batch)
                for _ in range(self.decoder_iteration):
                    for index in range(setting.num_message):
                        self.train_decoder(batch, index)

            for batch in dataset:
                image = batch[:1, :]
                messages = [np.random.choice(2, (1, setting.message_size)) for _ in range(setting.num_message)]
                decoded_image = self.generator(image, messages)
                decoded_messages = [self.decoders[index](decoded_image) for index in range(setting.num_message)]
                break

            cv2.imwrite(setting.sample_image, np.array((image[0]+1)*127.5))
            cv2.imwrite(setting.sample_decoded_image, np.array((decoded_image[0]+1)*127.5))

            if epoch_num % setting.save_iteration == 0:
                if self.generator_iteration:
                    self.generator.save(setting.GAN_pathes["generator"])
                if self.discriminator_iteration:
                    self.discriminator.save(setting.GAN_pathes["discriminator"])
                if self.decoder_iteration:
                    for index in range(setting.num_message):
                        self.decoders[index].save(setting.GAN_pathes["decoder"]+str(index))
            
            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Image Reconstruction Loss: " + str(self.generator_metric.result().numpy()))
            print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))
            for index in range(setting.num_message):
                print("Message Reconstruction " + str(index+1) + " Accuracy: " + str(self.decoders_metric[index].result().numpy()))

            print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
            print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))

    def test(self, dataset):
        dataset = dataset.take(setting.batch_size).batch(setting.batch_size, drop_remainder=True)

        self.generator_metric.reset_state()
        self.discriminator_metric.reset_state()
        for index in range(setting.num_message):
            self.decoders_metric[index].reset_state()

        for batch in dataset:
            image = batch[:1, :]
            messages = [np.random.choice(2, (1, setting.message_size)) for _ in range(setting.num_message)]
            decoded_image = self.generator(image, messages)
            decoded_messages = [self.decoders[index](decoded_image) for index in range(setting.num_message)]

            discriminator_ouput_true = self.discriminator(batch)
            discriminator_ouput_fake = self.discriminator(decoded_image)
            
            self.generator_metric.update_state(batch, decoded_image)

            self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
            self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

            for index in range(setting.num_message):
                self.decoders_metric[index].update_state(messages[index], decoded_messages[index])

            break

        cv2.imwrite(setting.sample_image, np.array((image[0]+1)*127.5))
        cv2.imwrite(setting.sample_decoded_image, np.array((decoded_image[0]+1)*127.5))

        print("Image Reconstruction Loss: " + str(self.generator_metric.result().numpy()))
        print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))
        for index in range(setting.num_message):
            print("Message Reconstruction " + str(index+1) + " Accuracy: " + str(self.decoders_metric[index].result().numpy()))

        print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
        print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))

