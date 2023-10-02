import tensorflow as tf

import setting
from layers import custom_conv2d, image_message_concatenation, xor_messages

class generator(tf.keras.Model):
    def __init__(self, key):
        super(generator, self).__init__()
        self.input_module = [
            custom_conv2d(32, 3),
        ]
        self.xor_layer = xor_messages(key)
        self.concat_layer = image_message_concatenation(setting.image_size)
        self.output_module = [
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
            tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', activation="tanh")
        ]

    def call(self, image, messages, training=False):

        middle_features = []
        middle_features.append(self.input_module[0](image, training))

        messages = messages.copy()
        for index in range(setting.num_message_channel):
            messages[index] = self.xor_layer(messages[index])
        middle_features[-1] = self.concat_layer(middle_features[-1], messages) # a, M

        middle_features.append(self.output_module[0](middle_features[-1], training)) # a, M, b
        middle_features.append(self.output_module[1](tf.concat(middle_features, -1), training)) # a, M, b, c
        middle_features.append(self.output_module[2](tf.concat(middle_features, -1))) # a, M, b, c, Eb
        return tf.clip_by_value(image+middle_features[-1], clip_value_min=-1, clip_value_max=1)

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = [
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
            tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')
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
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
            custom_conv2d(32, 3),
        ]
        self.output_module = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(setting.message_size)
        ]

    def call(self, x, training=False):
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
        return x
