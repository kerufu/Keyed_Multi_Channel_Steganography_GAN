import tensorflow as tf

import setting
from layers import custom_conv2d, custom_dense, image_label_concatenation, xor_messages

class generator(tf.keras.Model):
    def __init__(self, key):
        super(generator, self).__init__()
        self.input_module = [
            custom_conv2d(128, 5),
        ]
        self.xor_layer = xor_messages(key)
        self.concat_layer = image_label_concatenation(setting.image_size, 3)
        self.output_module = [
            custom_conv2d(128, 5),
            custom_conv2d(3, 3),
        ]

    def call(self, image, messages, training=False):
        for il in self.input_module:
            if "custom" in il.name:
                image = il(image, training)
            else:
                image = il(image)

        messages = messages.copy()
        for index in range(setting.num_message):
            messages[index] = self.xor_layer(messages[index])
        messages = tf.cast(tf.concat(messages, 1), tf.float32)
        messages = messages * 2 - 1
        image = self.concat_layer(image, messages)
        
        for ol in self.output_module:
            if "custom" in ol.name:
                image = ol(image, training)
            else:
                image = ol(image)
        return image

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.model = [
            custom_conv2d(64, 5, scale_down=True),
            custom_conv2d(128, 3, scale_down=True),
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
            custom_conv2d(64, 5, scale_down=True),
            custom_conv2d(128, 3, scale_down=True),
            custom_conv2d(256, 2, scale_down=True),
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
