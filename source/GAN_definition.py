import tensorflow as tf
import numpy

import setting
import layers

class generator(tf.keras.Model):
    def __init__(self, key, sum_residual=True):
        super(generator, self).__init__()
        self.hamming_layer = layers.HammingCode()
        self.sum_residual = sum_residual
        self.input_module = [
            layers.CustomConv2d(32, 3, reflect_padding=True),
        ]
        self.xor_layer = layers.XORMessages(key)
        self.concat_layer = layers.ImageMessageConcatenation(setting.image_size)
        self.output_module = [
            layers.CustomConv2d(32, 3, reflect_padding=True),
            layers.CustomConv2d(32, 3, reflect_padding=True),
            layers.ReflectRadding(3), 
            tf.keras.layers.Conv2D(3, 3, activation="tanh")
        ]

    def call(self, image, messages, enable_hamming=False, training=False):

        middle_features = []
        middle_features.append(self.input_module[0](image, training))

        messages = messages.copy()
        for index in range(setting.num_message_channel): # for isolation between recievers
            if enable_hamming:
                messages[index] = self.hamming_layer(messages[index])
            messages[index] = self.xor_layer(messages[index])
        middle_features[-1] = self.concat_layer(middle_features[-1], messages) # a, M

        middle_features.append(self.output_module[0](middle_features[-1], training)) # a, M, b
        middle_features.append(self.output_module[1](tf.concat(middle_features, -1), training)) # a, M, b, c
        middle_features.append(self.output_module[3](self.output_module[2](tf.concat(middle_features, -1)))) # a, M, b, c, Eb

        if self.sum_residual:
            return tf.clip_by_value(image+middle_features[-1], clip_value_min=-1, clip_value_max=1)
        else:
            return image + middle_features[-1] / 2

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = [
            layers.CustomConv2d(32, 3, clip_kernal=True),
            layers.CustomConv2d(32, 3, clip_kernal=True),
            layers.CustomConv2d(32, 3, clip_kernal=True),
            tf.keras.layers.Conv2D(1, 3, strides=1, padding='same', kernel_constraint=layers.ClipConstraint())
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return tf.reduce_mean(x, axis=[1, 2, 3])
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.input_module = [   
            layers.CustomConv2d(32, 3),
            layers.CustomConv2d(32, 3),
            layers.CustomConv2d(32, 3),
            layers.CustomConv2d(32, 3),
        ]
        self.output_module = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(setting.total_bit_size_per_channel)
        ]
        self.hamming_layer = layers.HammingCode(decode_mode=True)

    def call(self, x, enable_hamming=False, training=False):
        middle_features = []
        middle_features.append(self.input_module[0](x, training))
        middle_features.append(self.input_module[1](middle_features[0], training))
        middle_features.append(self.input_module[2](tf.concat(middle_features, -1), training))
        x = self.input_module[3](tf.concat(middle_features, -1), training)

        for layer in self.output_module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)

        if enable_hamming:
            x = tf.math.sigmoid(x)
            x = tf.math.round(x)
            x = self.hamming_layer(x)
            x = x - 0.5
        
        return x
