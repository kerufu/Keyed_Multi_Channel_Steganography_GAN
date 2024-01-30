import time

import tensorflow as tf
import numpy as np
from termcolor import cprint
import cv2
import image_similarity_measures.quality_metrics as image_metrics

import GAN_definition
import setting
import layers
import worker_pool

class GAN_worker():
    def __init__(self, key, generator_iteration=1, discriminator_iteration=5, decoder_iteration=1, authenticator_iteration=5, wgan=True) -> None:
        self.generator_iteration = generator_iteration
        self.discriminator_iteration = discriminator_iteration
        self.decoder_iteration = decoder_iteration
        self.authenticator_iteration = authenticator_iteration

        self.generator = GAN_definition.generator(key, clip_residual=False)
        self.discriminator = GAN_definition.discriminator()
        self.decoders = [GAN_definition.decoder() for _ in range(setting.num_message_channel)]
        self.authenticator = GAN_definition.authenticator()

        try:
            self.generator.load_weights(setting.GAN_pathes["generator"])
            self.discriminator.load_weights(setting.GAN_pathes["discriminator"])
            for index in range(setting.num_message_channel):
                self.decoders[index].load_weights(setting.GAN_pathes["decoder"]+str(index))
            self.authenticator.load_weights(setting.GAN_pathes["authenticator"])
            print("GAN model weight loaded")
        except:
            print("GAN model weight not found")

        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.decoder_opts = [tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay) for _ in range(setting.num_message_channel)]
        self.authenticator_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)

        self.generator_loss = tf.keras.losses.MeanSquaredError()
        if wgan:
            self.discriminator_loss = layers.WassersteinLoss()
        else:
            self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)
        self.decoder_loss = [tf.keras.losses.BinaryCrossentropy(from_logits=True) for _ in range(setting.num_message_channel)]
        self.authenticator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.compress_param = [
            int(cv2.IMWRITE_JPEG_LUMA_QUALITY), 85,
            int(cv2.IMWRITE_JPEG_CHROMA_QUALITY), 85,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
        ]
        self.compression_loss = tf.keras.losses.MeanSquaredError()

        self.generator_metric = tf.keras.metrics.MeanSquaredError()
        self.discriminator_metric = tf.keras.metrics.BinaryAccuracy(threshold=0) # for convenience, handle wgan score in the same way as logit, which is not actual
        self.decoders_metric = [tf.keras.metrics.BinaryAccuracy(threshold=0) for _ in range(setting.num_message_channel)]
        self.authenticator_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)

        self.compression_metric = tf.keras.metrics.MeanSquaredError()

    def compress_image(self, image):
        compressed_image = np.array(image)
        for _ in range(setting.jpeg_compression_iteration):
            for index in range(setting.batch_size):
                result, ci = cv2.imencode('.jpg', (compressed_image[index, :, :, :]+1)*127.5, self.compress_param)
                compressed_image[index, :, :, :] = cv2.imdecode((ci), 1) / 127.5 - 1
        return compressed_image

    def get_generator_loss(self, input_image, output_image_valid, output_image_invalid, messages, decoded_messages, discriminator_ouput_fake, authenticator_output_valid, authenticator_output_invalid):
        loss = self.generator_loss(input_image, output_image_valid) * setting.mse_weight_valid
        loss += self.generator_loss(input_image, output_image_invalid) * setting.mse_weight_invalid
        decoders_loss = 0
        for index in range(setting.num_message_channel):
            decoders_loss += self.decoder_loss[index](messages[index], decoded_messages[index])
        loss += decoders_loss * setting.decoder_weight
        loss += self.discriminator_loss(tf.ones_like(discriminator_ouput_fake), discriminator_ouput_fake) * setting.discriminator_weight
        if len(self.generator.losses) > 0 and setting.regularization_weight > 0:
            loss += tf.add_n(self.generator.losses) * setting.regularization_weight
        loss += self.authenticator_loss(tf.ones_like(authenticator_output_valid), authenticator_output_valid) * setting.authenticator_weight
        loss += self.authenticator_loss(tf.zeros_like(authenticator_output_invalid), authenticator_output_invalid) * setting.authenticator_weight

        if setting.jpeg_compression_loss_weight:
            compressed_image = self.compress_image(output_image_valid)
            loss += self.compression_loss(output_image_valid, compressed_image) * setting.jpeg_compression_loss_weight
            self.compression_metric.update_state(output_image_valid, compressed_image)

        return loss
    
    def get_discriminator_loss(self, target, output):
        loss = self.discriminator_loss(target, output)
        if len(self.discriminator.losses) > 0 and setting.regularization_weight > 0:
            loss += tf.add_n(self.discriminator.losses) * setting.regularization_weight
        return loss
    
    
    def get_decoder_loss(self, message, decoded_message, index):
        loss = self.decoder_loss[index](message, decoded_message)
        if len(self.decoders[index].losses) > 0 and setting.regularization_weight > 0:
            loss += tf.add_n(self.decoders[index].losses) * setting.regularization_weight
        return loss
    
    def get_authenticator_loss(self, authenticator_output_valid, authenticator_output_invalid):
        loss = self.authenticator_loss(tf.ones_like(authenticator_output_valid), authenticator_output_valid)
        loss += self.authenticator_loss(tf.zeros_like(authenticator_output_invalid), authenticator_output_invalid)
        if len(self.authenticator.losses) > 0 and setting.regularization_weight > 0:
            loss += tf.add_n(self.authenticator.losses) * setting.regularization_weight
        return loss
    
    def train_generator(self, batch, messages):
        def train_generator_step(batch, messages):
            with tf.GradientTape() as generator_tape:
                
                output_image_valid = self.generator(batch, messages, training=True)
                output_image_invalid = self.generator(batch, messages, training=True, random_key=True)

                decoded_messages = []
                for index in range(setting.num_message_channel):
                    decoded_messages.append(self.decoders[index](output_image_valid))
                
                discriminator_ouput_fake = self.discriminator(tf.concat([output_image_valid, output_image_invalid], axis=0))

                authenticator_output_valid = self.authenticator(output_image_valid)
                authenticator_output_invalid = self.authenticator(output_image_invalid)

                # tf.print(authenticator_output_valid)
                # tf.print(authenticator_output_invalid)

                generator_loss = self.get_generator_loss(batch, output_image_valid, output_image_invalid, messages, decoded_messages, discriminator_ouput_fake, authenticator_output_valid, authenticator_output_invalid)
            
            generator_gradient = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_opt.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))

            self.generator_metric.update_state(batch, output_image_valid)

        if setting.jpeg_compression_loss_weight <= 0:
            train_generator_step = tf.function(train_generator_step)

        return train_generator_step(batch, messages)

    @tf.function
    def train_discriminator(self, batch, messages):
        with tf.GradientTape() as discriminator_tape_true:
            discriminator_ouput_true = self.discriminator(batch, training=True)

            discriminator_loss_true = self.get_discriminator_loss(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)

        discriminator_gradient = discriminator_tape_true.gradient(discriminator_loss_true, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        with tf.GradientTape() as discriminator_tape_fake:
            output_image_valid = self.generator(batch, messages)
            output_image_invalid = self.generator(batch, messages, random_key=True)
            discriminator_ouput_fake = self.discriminator(tf.concat([output_image_valid, output_image_invalid], axis=0), training=True)

            discriminator_loss_fake = self.get_discriminator_loss(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

        discriminator_gradient = discriminator_tape_fake.gradient(discriminator_loss_fake, self.discriminator.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

        self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
        self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

    @tf.function
    def train_decoder(self, batch, messages):
        for index in range(setting.num_message_channel):
            with tf.GradientTape() as decoder_tape:
                output_image = self.generator(batch, messages)
                decoded_message = self.decoders[index](output_image, training=True)
                decoder_loss = self.get_decoder_loss(messages[index], decoded_message, index)
            decoder_gradient = decoder_tape.gradient(decoder_loss, self.decoders[index].trainable_variables)
            self.decoder_opts[index].apply_gradients(zip(decoder_gradient, self.decoders[index].trainable_variables))
            self.decoders_metric[index].update_state(messages[index], decoded_message)

    @tf.function
    def train_authenticator(self, batch, messages):
        with tf.GradientTape() as authenticator_tape:
            output_image_valid = self.generator(batch, messages)
            output_image_invalid = self.generator(batch, messages, random_key=True)

            authenticator_output_valid = self.authenticator(output_image_valid, training=True)
            authenticator_output_invalid = self.authenticator(output_image_invalid, training=True)
            
            # tf.print(authenticator_output_valid)
            # tf.print(authenticator_output_invalid)

            authenticator_loss = self.get_authenticator_loss(authenticator_output_valid, authenticator_output_invalid)

        authenticator_gradient = authenticator_tape.gradient(authenticator_loss, self.authenticator.trainable_variables)
        self.authenticator_opt.apply_gradients(zip(authenticator_gradient, self.authenticator.trainable_variables))

        self.authenticator_metric.update_state(tf.ones_like(authenticator_output_valid), authenticator_output_valid)
        self.authenticator_metric.update_state(tf.zeros_like(authenticator_output_invalid), authenticator_output_invalid)

    def train(self, epoch):
        msg_acc_max = 0
        auth_acc_max = 0
        mse_min = np.inf

        for epoch_num in range(epoch):
            start = time.time()
            self.generator_metric.reset_state()
            self.discriminator_metric.reset_state()
            for index in range(setting.num_message_channel):
                self.decoders_metric[index].reset_state()
            self.authenticator_metric.reset_state()
            if setting.jpeg_compression_loss_weight:
                self.compression_metric.reset_state()
            
            for batch in worker_pool.dw.dataset:
                for _ in range(self.discriminator_iteration):
                    messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
                    self.train_discriminator(batch, messages)
                for _ in range(self.generator_iteration):
                    messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
                    self.train_generator(batch, messages)
                for _ in range(self.decoder_iteration):
                    messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
                    self.train_decoder(batch, messages)
                for _ in range(self.authenticator_iteration):
                    messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
                    self.train_authenticator(batch, messages)

            messages = [np.random.choice(2, (1, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            encoded_image = self.generator(batch, messages)
            decoded_messages = [self.decoders[index](encoded_image) for index in range(setting.num_message_channel)]

            cv2.imwrite(setting.sample_image, np.array((batch[0]+1)*127.5))
            cv2.imwrite(setting.sample_encoded_image, np.array((encoded_image[0]+1)*127.5))
            
            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')
            mse = self.generator_metric.result().numpy()
            print("Image Reconstruction Loss: " + str(mse))
            if setting.jpeg_compression_loss_weight:
                print("Image Compression Loss: " + str(self.compression_metric.result().numpy()))
            print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))

            msg_acc_list = []
            for index in range(setting.num_message_channel):
                msg_acc = self.decoders_metric[index].result().numpy()
                msg_acc_list.append(msg_acc)
                print("Message " + str(index+1) + " Reconstruction Accuracy: " + str(msg_acc))
            msg_acc_mean = np.mean(msg_acc_list)
            auth_acc = self.authenticator_metric.result().numpy()

            print("Message Reconstruction Average Accuracy: " + str(msg_acc_mean))
            print("Authenticator Accuracy: " + str(auth_acc))
            print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
            print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))

            save_con1 = msg_acc_mean > msg_acc_max
            save_con2 = msg_acc_mean > msg_acc_max - 0.0001
            save_con3 = save_con2 and (auth_acc >= auth_acc_max)
            save_con2 = save_con2 and (mse < mse_min + 0.0001)

            if save_con1:
                msg_acc_max = msg_acc_mean
            elif save_con2:
                mse_min = mse
            elif save_con3:
                auth_acc_max = auth_acc

            if save_con1 or save_con2 or save_con3:
                if self.generator_iteration:
                    self.generator.save(setting.GAN_pathes["generator"])
                if self.discriminator_iteration:
                    self.discriminator.save(setting.GAN_pathes["discriminator"])
                if self.decoder_iteration:
                    for index in range(setting.num_message_channel):
                        self.decoders[index].save(setting.GAN_pathes["decoder"]+str(index))
                if self.authenticator_iteration:
                    self.authenticator.save(setting.GAN_pathes["authenticator"])
                print("Model Saved")

    def image_quantization(self, image):
        image += 1
        image *= 127.5
        image = tf.floor(image)
        image /= 127.5
        image -= 1
        return image

    def evaluate(self, coding_mode=0):
        self.generator_metric.reset_state()
        self.discriminator_metric.reset_state()
        for index in range(setting.num_message_channel):
            self.decoders_metric[index].reset_state()
        self.authenticator_metric.reset_state()
        if setting.jpeg_compression_loss_weight:
            self.compression_metric.reset_state()

        enable_hamming = False
        enable_character_mapping = False
        if coding_mode == 1:
            enable_hamming = True
        elif coding_mode == 2:
            enable_character_mapping = True

        for batch in worker_pool.dw.dataset:
            if enable_hamming:
                messages = [np.random.choice(2, (setting.batch_size, setting.data_bit_size_per_channel)) for _ in range(setting.num_message_channel)]
            elif enable_character_mapping:
                messages = []
                for _ in range(setting.num_message_channel):
                    m = np.random.choice(list(worker_pool.cm.mapping_table.keys()), (setting.batch_size, setting.num_of_window_per_channel))
                    m = np.expand_dims(m, axis=2)
                    m = np.apply_along_axis(lambda key: worker_pool.cm.mapping_table[int(key)], axis=2, arr=m)
                    m = tf.concat(m, axis=-1)
                    m = np.array(m)
                    m = m.reshape((-1, setting.total_bit_size_per_channel)).astype(int)
                    messages.append(m)
            else:
                messages = [np.random.choice(2, (setting.batch_size, setting.total_bit_size_per_channel)) for _ in range(setting.num_message_channel)]

            encoded_image_valid = self.generator(batch, messages, enable_hamming=enable_hamming)
            encoded_image_invalid = self.generator(batch, messages, enable_hamming=enable_hamming, random_key=True)

            # encoded_image_valid = self.image_quantization(encoded_image_valid)
            # encoded_image_invalid = self.image_quantization(encoded_image_invalid)

            decoded_messages = [self.decoders[index](encoded_image_valid, enable_hamming=enable_hamming) for index in range(setting.num_message_channel)]
            
            if enable_character_mapping:
                for index in range(setting.num_message_channel):
                    decoded_messages[index] = tf.math.sigmoid(decoded_messages[index])
                    decoded_messages[index] = tf.math.round(decoded_messages[index])
                    decoded_messages[index] = np.array(decoded_messages[index])
                    decoded_messages[index] = decoded_messages[index].reshape((-1, setting.num_of_window_per_channel, setting.coding_window_size))
                    decoded_messages[index] = np.apply_along_axis(worker_pool.cm.bits_matching, axis=2, arr=decoded_messages[index])
                    decoded_messages[index] = decoded_messages[index].astype(np.float32)
                    decoded_messages[index] = decoded_messages[index].reshape((-1, setting.total_bit_size_per_channel))
                    decoded_messages[index] = decoded_messages[index] - 0.5

            discriminator_ouput_true = self.discriminator(batch)
            discriminator_ouput_fake = self.discriminator(encoded_image_valid)

            authenticator_output_valid = self.authenticator(encoded_image_valid)
            authenticator_output_invalid = self.authenticator(encoded_image_invalid)

            # tf.print(authenticator_output_valid)
            # tf.print(authenticator_output_invalid)
                    
            self.generator_metric.update_state(batch, encoded_image_valid)
            if setting.jpeg_compression_loss_weight:
                compressed_image = self.compress_image(encoded_image_valid)
                self.compression_metric.update_state(encoded_image_valid, compressed_image)

            self.discriminator_metric.update_state(tf.ones_like(discriminator_ouput_true), discriminator_ouput_true)
            self.discriminator_metric.update_state(tf.zeros_like(discriminator_ouput_fake), discriminator_ouput_fake)

            self.authenticator_metric.update_state(tf.ones_like(authenticator_output_valid), authenticator_output_valid)
            self.authenticator_metric.update_state(tf.zeros_like(authenticator_output_invalid), authenticator_output_invalid)

            for index in range(setting.num_message_channel):
                self.decoders_metric[index].update_state(messages[index], decoded_messages[index])

        cv2.imwrite(setting.sample_image, np.array((batch[0]+1)*127.5))
        cv2.imwrite(setting.sample_encoded_image, np.array((encoded_image_valid[0]+1)*127.5))

        print("Image Reconstruction Loss: " + str(self.generator_metric.result().numpy()))

        def get_image_metric(metric, set_max_p=False):
            result = 0
            for index in range(setting.batch_size):
                if set_max_p:
                    result += metric(np.array(batch[index]), np.array(encoded_image_valid[index]), max_p=2)
                else:
                    result += metric(np.array(batch[index]), np.array(encoded_image_valid[index]))
            return result / setting.batch_size

        print("Image SSIM: " + str(get_image_metric(image_metrics.ssim, set_max_p=True)))
        print("Image PSNR: " + str(get_image_metric(image_metrics.psnr, set_max_p=True)))
        print("Image FSIM: " + str(get_image_metric(image_metrics.fsim)))
        print("Image SRE: " + str(get_image_metric(image_metrics.sre)))
        print("Image SAM: " + str(get_image_metric(image_metrics.sam)))
        print("Image UIQ: " + str(get_image_metric(image_metrics.uiq)))

        if setting.jpeg_compression_loss_weight:
            print("Image Compression Loss: " + str(self.compression_metric.result().numpy()))
        print("Discriminator Accuracy: " + str(self.discriminator_metric.result().numpy()))
        
        acc_list = []
        for index in range(setting.num_message_channel):
            acc = self.decoders_metric[index].result().numpy()
            acc_list.append(acc)
            print("Message " + str(index+1) + " Reconstruction Accuracy: " + str(acc))
        acc_mean = np.mean(acc_list)

        print("Message Reconstruction Average Accuracy: " + str(acc_mean))
        print("Authenticator Accuracy: " + str(self.authenticator_metric.result().numpy()))
        print("Sample Messages: " + str(np.array(messages[0][0])[:10]))
        print("Sample Decoded Messages: " + str(np.array(decoded_messages[0][0])[:10]))

    def plot(self):
        tf.keras.utils.plot_model(self.generator.model(), "generator.png", show_shapes=True)
        tf.keras.utils.plot_model(self.discriminator.model(), "discriminator.png", show_shapes=True)
        for index in range(setting.num_message_channel):
            tf.keras.utils.plot_model(self.decoders[index].model(), "decoder_"+str(index) +".png", show_shapes=True)
        tf.keras.utils.plot_model(self.authenticator.model(), "authenticator.png", show_shapes=True)