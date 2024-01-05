import tensorflow as tf
import numpy as np

import setting
import layers

class generator(tf.keras.Model):
    def __init__(self, key, clip_residual=False):
        super(generator, self).__init__()
        self.hamming_layer = layers.HammingCode()
        self.clip_residual = clip_residual
        self.input_module = [
            layers.CustomConv2D(setting.num_conv_channel, reflect_padding=True)
        ]
        self.xor_layer = layers.EncrypteMessage(key)
        self.concat_layer = layers.ImageMessageConcatenation()
        self.output_module = [
            layers.CustomConv2D(setting.num_conv_channel, reflect_padding=True),
            layers.CustomConv2D(setting.num_conv_channel, reflect_padding=True),
            layers.CustomConv2D(3, batch_normalization=False, reflect_padding=True, activation="htanh")
        ]

    def call(self, image, messages, enable_hamming=False, training=False):

        input_feature = tf.identity(image)
        for im in self.input_module:
            input_feature = im(image, training)

        messages = messages.copy()
        for index in range(setting.num_message_channel): # for isolation between recievers
            if enable_hamming:
                messages[index] = self.hamming_layer(messages[index])
            messages[index] = self.xor_layer(messages[index])

        middle_features = []
        middle_features.append(self.concat_layer(input_feature, messages)) # a, M
        middle_features.append(self.output_module[0](middle_features[-1], training)) # a, M, b
        middle_features.append(self.output_module[1](tf.concat(middle_features, -1), training)) # a, M, b, c

        output_feature = self.output_module[2](tf.concat(middle_features, -1), training)
        output_feature += image

        if self.clip_residual:
            return tf.clip_by_value(output_feature, clip_value_min=-1, clip_value_max=1)
        else:
            return output_feature / 2
        
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        message_input = [tf.keras.Input(shape=(setting.total_bit_size_per_channel,), dtype='int64', name='message_input_'+str(index)) for index in range(setting.num_message_channel)]
        return tf.keras.Model(inputs=[image_input, message_input], outputs=self.call(image_input, message_input))

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.module = [
            layers.CustomConv2D(setting.num_conv_channel, clip_kernal=True),
            layers.CustomConv2D(setting.num_conv_channel, clip_kernal=True),
            layers.CustomConv2D(setting.num_conv_channel, clip_kernal=True),
            layers.CustomConv2D(1, batch_normalization=False, scale_down_mode=1, clip_kernal=True, activation="linear")
        ]

    def call(self, x, training=False):
        for layer in self.module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return tf.reduce_mean(x, axis=[1, 2, 3])
    
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        return tf.keras.Model(inputs=[image_input], outputs=self.call(image_input))
    
class decoder(tf.keras.Model):
    def __init__(self, conv_flatten=False):
        super(decoder, self).__init__()
        self.input_module = [   
            layers.CustomConv2D(setting.num_conv_channel),
            layers.CustomConv2D(setting.num_conv_channel),
            layers.CustomConv2D(setting.num_conv_channel),
            layers.CustomConv2D(setting.num_conv_channel)
        ]
        if conv_flatten:
            self.output_module = [
                layers.CustomFlatten(setting.total_bit_size_per_channel)
            ]
        else:
            self.output_module = [
                tf.keras.layers.Flatten(),
                layers.CustomDense(setting.total_bit_size_per_channel, batch_normalization=False, activation="linear")
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
    
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        return tf.keras.Model(inputs=[image_input], outputs=self.call(image_input))


class authenticator(tf.keras.Model):
    def __init__(self):
        super(authenticator, self).__init__()
        self.module = [
            layers.CustomConv2D(setting.num_conv_channel),
            layers.CustomFlatten(setting.num_conv_channel),
            layers.CustomDense(1, batch_normalization=False, activation="linear")
        ]

    def call(self, x, training=False):
        for layer in self.module:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        return tf.keras.Model(inputs=[image_input], outputs=self.call(image_input))