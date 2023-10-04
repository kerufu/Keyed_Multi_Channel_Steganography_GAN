import tensorflow as tf

import setting

class WassersteinLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def call(self, y_true, y_pred):
        y_true = y_true * 2 - 1
        return -tf.math.reduce_mean(y_true*y_pred)

class ClipConstraint(tf.keras.constraints.Constraint):
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

class xor_messages(tf.keras.layers.Layer):

    def __init__(self, xor_key):
        super(xor_messages, self).__init__()
        self.xor_key = xor_key

    def call(self, messages):
        return tf.bitwise.bitwise_xor(messages, self.xor_key)

class image_message_concatenation(tf.keras.layers.Layer):
    
    def __init__(self, image_size):
        super(image_message_concatenation, self).__init__()
        self.image_size = image_size
        self.arbitary_message_length = not isinstance(setting.message_bit_per_pixel, int)
        if self.arbitary_message_length:
            self.message_layer = custom_dense(self.image_size*self.image_size*(int(setting.message_bit_per_pixel)+1))

    def call(self, image, messages):
        message = tf.cast(tf.concat(messages, 1), tf.float32)
        message = message * 2 - 1
        if self.arbitary_message_length:
            message = self.message_layer(message)
            message = tf.reshape(message, (-1, self.image_size, self.image_size, int(setting.message_bit_per_pixel)+1))
        else:
            message = tf.reshape(message, (-1, self.image_size, self.image_size, setting.message_bit_per_pixel))
        return tf.concat([image, message], -1)

class reflect_padding_layer(tf.keras.layers.Layer): # O=[(W−K+P)/S]+1
    def __init__(self, kernel_size):
        super(reflect_padding_layer, self).__init__()
        pad = kernel_size - 1
        self.upper_pad = pad // 2
        self.lower_pad = pad - self.upper_pad

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.upper_pad, self.lower_pad], [self.upper_pad, self.lower_pad], [0, 0]], 'REFLECT')

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, reflect_padding=False, scale_down_mode=0, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_conv2d, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if reflect_padding:
            if scale_down_mode == 0:
                self.model = [
                    reflect_padding_layer(kernel_size),
                    tf.keras.layers.Conv2D(num_channel, kernel_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 1:
                self.model = [
                    reflect_padding_layer(kernel_size),
                    tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 2:
                self.model = [
                    reflect_padding_layer(kernel_size),
                    tf.keras.layers.Conv2D(num_channel, kernel_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                    tf.keras.layers.MaxPool2D(),
                ]
        else:
            if scale_down_mode == 0:
                self.model = [
                    tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 1:
                self.model = [
                    tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 2:
                self.model = [
                    tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                    tf.keras.layers.MaxPool2D(),
                ]

        self.model += [
            tf.keras.layers.Activation(activation),
            tf.keras.layers.BatchNormalization()
        ]
        if dropout:
            self.model.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class custom_dense(tf.keras.layers.Layer):

    def __init__(self, output_size, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_dense, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.model = [
            tf.keras.layers.Dense(output_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.BatchNormalization()
        ]
        if dropout:
            self.model.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x