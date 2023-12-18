import tensorflow as tf
import numpy as np

import setting

class HammingCode(tf.keras.layers.Layer):
    def __init__(self, decode_mode=False):
        super(HammingCode, self).__init__()
        self.decode_mode = decode_mode

    def encode(self, data_bits):
        bit_stream = []
        data_index = 0
        parity_ints = np.array([0]*setting.batch_size)
        for index in range(setting.coding_window_size-1):
            if index in setting.parity_indexes:
                bit_stream.append(np.array([-1]*setting.batch_size))
            else:
                bit_vector = data_bits[:, data_index]
                parity_ints = tf.bitwise.bitwise_xor(parity_ints, np.where(bit_vector==1, index+1, 0))
                bit_stream.append(bit_vector)
                data_index += 1

        parity_bits = []
        for index in range(setting.batch_size):
            bits = np.binary_repr(parity_ints[index], width=setting.parity_bit_size_per_window)
            bits = np.fromstring(bits,'u1') - ord('0')
            parity_bits.append(bits)

        bit_stream = np.array(bit_stream).T
        parity_bits = np.array(parity_bits)[:, ::-1]

        for index in range(setting.parity_bit_size_per_window):
            bit_stream[:, setting.parity_indexes[index]] = parity_bits[:, index]
        window_parity = np.expand_dims(np.logical_xor.reduce(bit_stream, axis=1), axis=1)
        bit_stream = tf.concat(values=[window_parity, bit_stream], axis=1)
        bit_stream = np.array(bit_stream).astype(int)

        return bit_stream, parity_bits
    
    def decode(self, bit_stream):
        window_parity = bit_stream[:, 0]
        bit_stream = bit_stream[:, 1:]
        recieved_parity_bits = []
        data_bits = []
        for index in range(setting.coding_window_size-1):
            if index in setting.parity_indexes:
                recieved_parity_bits.append(bit_stream[:, index])
            else:
                data_bits.append(bit_stream[:, index])
        data_bits = np.array(data_bits).T
        recieved_parity_bits = np.array(recieved_parity_bits).T
        computed_bit_stream, computed_parity_bits = self.encode(data_bits)
        possibly_single_error = np.where(window_parity!=computed_bit_stream[:, 0], True, False)

        distance = [0] * setting.batch_size
        distance = np.array(distance)
        temp_distance = [0] * setting.batch_size
        temp_distance = np.array(temp_distance)

        for index in range(setting.parity_bit_size_per_window):
            temp_distance += np.where((recieved_parity_bits[:, index]==computed_parity_bits[:, index]), 0, setting.parity_indexes[index]+1)
            if np.all(np.less(temp_distance, setting.data_bit_size_per_window)):
                distance = temp_distance.copy()
        distance = np.where(possibly_single_error, distance, 0)
        for pi in setting.parity_indexes:
            distance = np.where(distance==pi+1, 0, distance)
        
        if not np.array_equal(distance, 0):
            offset = [1] * setting.batch_size
            offset = np.array(offset)
            for pi in setting.parity_indexes:
                offset += np.where(pi+1<distance, 1, 0)
            distance -= offset
            for index in range(setting.batch_size):
                if distance[index] >= 0:
                    data_bits[index, distance[index]] = 1 - data_bits[index, distance[index]]

        return data_bits

    def call(self, bit_stream):
        result = []

        if self.decode_mode:
            for index in range(setting.num_of_window_per_channel):
                bs = bit_stream[:, setting.coding_window_size*index:setting.coding_window_size*(index+1)]
                db = self.decode(bs)
                result.append(db)
            if setting.residual_bits_size:
                result.append(bit_stream[:, -setting.residual_bits_size:])
        else:
            for index in range(setting.num_of_window_per_channel):
                bs = bit_stream[:, setting.data_bit_size_per_window*index:setting.data_bit_size_per_window*(index+1)]
                ebs, pb = self.encode(bs)
                result.append(ebs)
            if setting.residual_bits_size:
                result.append(bit_stream[:, -setting.residual_bits_size:])
        
        result = tf.concat(result, axis=1)
        result = np.array(result).astype(int)
        return result

class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = y_true * 2 - 1
        return -tf.math.reduce_mean(y_true*y_pred)

class HardTanh(tf.keras.layers.Layer):
    def call(self, x):
        return tf.minimum(tf.maximum(x, -1), 1)

class HardSwish(tf.keras.layers.Layer):
    def call(self, x):
        x = tf.where(tf.less_equal(x, -3), 0.0, x)
        x = tf.where(tf.logical_and(tf.greater(x, -3), tf.less(x, 3)), (x*(x+3))/6, x)
        return x

class ClipConstraint(tf.keras.constraints.Constraint):
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

class EncrypteMessage(tf.keras.layers.Layer):
    def __init__(self, key, xor=True):
        super(EncrypteMessage, self).__init__()
        self.xor = xor

        self.key = tf.repeat(key, repeats=setting.image_size//setting.key_size, axis=0)
        self.key = tf.repeat([self.key], repeats=setting.image_size//setting.num_message_channel, axis=0)
        self.key = tf.repeat([self.key], repeats=setting.message_bit_per_pixel, axis=0)
        self.key = tf.reshape(self.key, [-1])

        if not self.xor:
            self.key = tf.repeat([self.key], repeats=setting.batch_size, axis=0)
            self.key = tf.cast(self.key, tf.float32) - 0.5

    def call(self, message):
        if self.xor:
            message = tf.bitwise.bitwise_xor(message, self.key)

        message = tf.cast(message, tf.float32)
        message = message * 2 - 1

        if not self.xor:
            message = message + self.key

        return message

class MaskFeature(tf.keras.layers.Layer):
    def __init__(self, mask):
        super(MaskFeature, self).__init__()
        self.mask = tf.repeat(mask, repeats=setting.image_size//setting.key_size)
        self.mask = tf.repeat([self.mask], repeats=setting.image_size)
        self.mask = tf.repeat([self.mask], repeats=setting.batch_size, axis=0)
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.mask -= 0.5

    def call(self, feature):
        feature = feature * 2 - 1
        feature = self.mask + feature
        return feature

class ImageMessageConcatenation(tf.keras.layers.Layer):
    
    def __init__(self):
        super(ImageMessageConcatenation, self).__init__()
        if isinstance(setting.message_bit_per_pixel, int):
            self.message_module = [
                tf.keras.layers.Reshape((setting.image_size, setting.image_size, setting.message_bit_per_pixel))
            ]
        else:
            self.message_module = [
                CustomDense(setting.image_size*setting.image_size*(int(setting.message_bit_per_pixel)+1)),
                tf.keras.layers.Reshape((setting.image_size, setting.image_size, (int(setting.message_bit_per_pixel)+1)))
            ]

    def call(self, image, messages):
        messages = tf.concat(messages, 1)
        for mm in self.message_module:
            messages = mm(messages)
        return tf.concat([image, messages], -1)

class ReflectRadding(tf.keras.layers.Layer): # O=[(W−K+P)/S]+1
    def __init__(self, kernel_size):
        super(ReflectRadding, self).__init__()
        pad = kernel_size - 1
        self.upper_pad = pad // 2
        self.lower_pad = pad - self.upper_pad

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.upper_pad, self.lower_pad], [self.upper_pad, self.lower_pad], [0, 0]], 'REFLECT')

class CustomConv2D(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, reflect_padding=False, scale_down_mode=0, clip_kernal=False, activation="hswish", depthwise_seperable=False):
        super(CustomConv2D, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if depthwise_seperable:
            Conv2D = tf.keras.layers.SeparableConv2D
        else:
            Conv2D = tf.keras.layers.Conv2D

        if reflect_padding:
            if scale_down_mode == 0:
                self.module = [
                    ReflectRadding(kernel_size),
                    Conv2D(num_channel, kernel_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 1:
                self.module = [
                    ReflectRadding(kernel_size),
                    Conv2D(num_channel, kernel_size, strides=2, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 2:
                self.module = [
                    ReflectRadding(kernel_size),
                    Conv2D(num_channel, kernel_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                    tf.keras.layers.MaxPool2D(),
                ]
        else:
            if scale_down_mode == 0:
                self.module = [
                    Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 1:
                self.module = [
                    Conv2D(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 2:
                self.module = [
                    Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                    tf.keras.layers.MaxPool2D(),
                ]

        self.module.append(tf.keras.layers.BatchNormalization())
        if activation == "hswish":
            self.module.append(HardSwish())
        else:
            self.module.append(tf.keras.layers.Activation(activation))
        if setting.dropout_ratio:
            self.module.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.module:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x

class CustomDense(tf.keras.layers.Layer):

    def __init__(self, output_size, clip_kernal=False, activation="leaky_relu"):
        super(CustomDense, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.module = [
            tf.keras.layers.Dense(output_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation)
        ]
        if setting.dropout_ratio:
            self.module.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.module:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x