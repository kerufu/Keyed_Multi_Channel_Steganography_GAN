import time

import tensorflow as tf
import numpy as np
from termcolor import cprint
import cv2

import worker_factory
import AE_definition
import setting

class AE_worker():
    def __init__(self, key):
        self.encoder = AE_definition.encoder()
        self.decoder = AE_definition.decoder(key)
        self.generator = worker_factory.ganw.generator

        try:
            self.encoder.load_weights(setting.AE_pathes["encoder"])
            self.decoder.load_weights(setting.AE_pathes["decoder"])
            print("AE model weight loaded")
        except:
            print("AE model weight not found")

        self.encoder_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.decoder_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)

        self.reconstruction_loss = tf.keras.losses.MeanSquaredError()
        self.randomness_loss = tf.keras.losses.KLDivergence()

        self.reconstruction_metric = tf.keras.metrics.MeanSquaredError()
        self.randomness_metric = tf.keras.metrics.BinaryAccuracy()

    def get_encoder_loss(self, target_image, output_image, target_feature, feature):
        loss = self.reconstruction_loss(target_image, output_image)
        loss += self.randomness_loss(target_feature, feature) * setting.regularization_weight
        loss += tf.add_n(self.encoder.losses) * setting.regularization_weight
        return loss
    
    def get_decoder_loss(self, target_image, output_image):
        loss = self.reconstruction_loss(target_image, output_image)
        loss += tf.add_n(self.decoder.losses) * setting.regularization_weight
        return loss

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as encoder_tape:
            with tf.GradientTape() as decoder_tape:
                messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
                input_image = self.generator(batch, messages)
                
                feature = self.encoder(input_image, training=True)
                target_feature = tf.random.uniform(tf.shape(feature))
                output_image = self.decoder(feature)

                encoder_loss = self.get_encoder_loss(batch, output_image, target_feature, feature)
                decoder_loss = self.get_decoder_loss(batch, output_image)
        
        encoder_gradient = encoder_tape.gradient(encoder_loss, self.encoder.trainable_variables)
        self.encoder_opt.apply_gradients(zip(encoder_gradient, self.encoder.trainable_variables))

        decoder_gradient = decoder_tape.gradient(decoder_loss, self.decoder.trainable_variables)
        self.decoder_opt.apply_gradients(zip(decoder_gradient, self.decoder.trainable_variables))

        self.reconstruction_metric.update_state(batch, output_image)
        self.randomness_metric.update_state(target_feature, feature)

    def train(self, epoch, dataset):
        dataset = dataset.take(setting.batch_size)
        dataset = dataset.shuffle(dataset.cardinality()//setting.shuffle_buffer_size_divider, reshuffle_each_iteration=True).batch(setting.batch_size, drop_remainder=True)

        recons_loss_min = np.inf

        for epoch_num in range(epoch):
            start = time.time()
            self.reconstruction_metric.reset_state()
            self.randomness_metric.reset_state()
            
            for batch in dataset:
                self.train_step(batch)

            image = batch[:1, :]
            messages = [np.random.choice(2, (1, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            input_image = self.generator(image, messages)
            feature = self.encoder(input_image)
            decoded_image = self.decoder(feature)

            cv2.imwrite(setting.sample_image, np.array((image[0]+1)*127.5))
            cv2.imwrite(setting.sample_reconstructed_image, np.array((decoded_image[0]+1)*127.5))

            recons_loss = self.randomness_metric.result().numpy()

            if recons_loss < recons_loss_min:
                recons_loss_min = recons_loss
                self.encoder.save(setting.AE_pathes["encoder"])
                self.decoder.save(setting.AE_pathes["decoder"])

            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Image Reconstruction Loss: " + str(self.reconstruction_metric.result().numpy()))
            print("Randomness Loss: " + str(self.randomness_metric.result().numpy()))

    def encode(self, image):
        image = np.array([worker_factory.dw.preprocess_image(image)])
        return self.encoder(image)[0]
