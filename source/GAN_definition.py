import tensorflow as tf

import setting
import layers

class generator(tf.keras.Model):
    def __init__(self, key, clip_residual=False):
        super(generator, self).__init__()
        self.hamming_layer = layers.HammingCode()
        self.clip_residual = clip_residual
        self.input_module = [
            layers.CustomConv2D(32, 3, reflect_padding=True),
        ]
        self.xor_layer = layers.EncrypteMessage(key)
        self.concat_layer = layers.ImageMessageConcatenation()
        self.output_module = [
            layers.CustomConv2D(32, 3, reflect_padding=True, depthwise_seperable=True),
            layers.CustomConv2D(32, 3, reflect_padding=True, activation="leaky_relu"),
            layers.ReflectRadding(3),
            tf.keras.layers.Conv2D(3, 3),
            layers.HardTanh()
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

        output_feature = self.output_module[3](self.output_module[2](tf.concat(middle_features, -1)))
        output_feature = self.output_module[4](output_feature)
        if self.clip_residual:
            return tf.clip_by_value(image+output_feature, clip_value_min=-1, clip_value_max=1)
        else:
            return (image + output_feature) / 2
        
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        message_input = [tf.keras.Input(shape=(setting.total_bit_size_per_channel,), dtype='int64', name='message_input_'+str(index)) for index in range(setting.num_message_channel)]
        return tf.keras.Model(inputs=[image_input, message_input], outputs=self.call(image_input, message_input))

class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()
        self.module = [
            layers.CustomConv2D(32, 3, scale_down_mode=1, clip_kernal=True, depthwise_seperable=True),
            layers.CustomConv2D(32, 3, scale_down_mode=1, clip_kernal=True, activation="leaky_relu"),
            layers.CustomConv2D(32, 3, scale_down_mode=1, clip_kernal=True, activation="leaky_relu"),
            tf.keras.layers.Conv2D(1, 3, kernel_constraint=layers.ClipConstraint())
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
    def __init__(self):
        super(decoder, self).__init__()
        self.input_module = [   
            layers.CustomConv2D(32, 3, depthwise_seperable=True),
            layers.CustomConv2D(32, 3, depthwise_seperable=True),
            layers.CustomConv2D(32, 3, depthwise_seperable=True),
            layers.CustomConv2D(32, 3)
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
    
    def model(self):
        image_input = tf.keras.Input(shape=(setting.image_size, setting.image_size, 3), dtype='float32', name='image_input')
        return tf.keras.Model(inputs=[image_input], outputs=self.call(image_input))
