import tensorflow as tf

import layers
import setting

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.module = [
            layers.CustomConv2D(16, 3, reflect_padding=True),
            layers.CustomConv2D(32, 3, reflect_padding=True),
            layers.ReflectRadding(3), 
            tf.keras.layers.Conv2D(setting.AE_feature_size, 3, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1L2(), activity_regularizer=tf.keras.regularizers.L1L2())
        ]

    def call(self, x, training=False):
        for layer in self.module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x

class decoder(tf.keras.Model):
    def __init__(self, key):
        super(decoder, self).__init__()
        self.output_module = [
            layers.MaskFeature(key),
            layers.CustomConv2D(32, 3, reflect_padding=True),
            layers.CustomConv2D(16, 3, reflect_padding=True),
            layers.ReflectRadding(3), 
            tf.keras.layers.Conv2D(3, 3, activation="tanh")
        ]

    def call(self, x, training=False):
        for layer in self.output_module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
