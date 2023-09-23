import tensorflow as tf

import setting
from layers import custom_conv2d, custom_conv2dtp, custom_dense, image_label_concatenation

class generator(tf.keras.Model):
    def __init__(self, key):
        super(generator, self).__init__()
        self.encoder = [
            custom_conv2d(64, 3, scale_down=True),
            custom_conv2d(128, 5, scale_down=True),
            custom_conv2d(256, 3, scale_down=True),
            custom_conv2d(512, 3, scale_down=True),
        ]
        self.xor_layer = xor_messages(key)
        self.concat_layer = image_label_concatenation(setting.image_size//16, 512)
        self.decoder = [
            custom_conv2dtp(256, 3, scale_up=True),
            custom_conv2dtp(128, 3, scale_up=True),
            custom_conv2dtp(64, 5, scale_up=True),
            tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation="tanh")
        ]

    def call(self, image, messages, training=False):
        for el in self.encoder:
            if "custom" in el.name:
                image = el(image, training)
            else:
                image = el(image)

        for index in range(setting.num_message):
            messages[index] = self.xor_layer(messages[index])
        messages = tf.cast(tf.concat(messages, 1), tf.float32)
        messages = messages * 2 - 1
        image = self.concat_layer(image, messages)
        for dl in self.decoder:
            if "custom" in dl.name:
                image = dl(image, training)
            else:
                image = dl(image)
        return image

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = [
            custom_conv2d(64, 3, scale_down=True),
            custom_conv2d(128, 5, scale_down=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = [
            custom_conv2d(64, 3, scale_down=True),
            custom_conv2d(128, 5, scale_down=True),
            custom_conv2d(256, 3, scale_down=True),
            custom_conv2d(512, 3, scale_down=True),
            tf.keras.layers.Flatten(),
            custom_dense(setting.message_size*4),
            tf.keras.layers.Dense(setting.message_size)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
