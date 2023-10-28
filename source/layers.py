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

        parity_bits = []
        data_bits = []
        for index in range(setting.coding_window_size-1):
            if index in setting.parity_indexes:
                parity_bits.append(bit_stream[:, index])
            else:
                data_bits.append(bit_stream[:, index])

        data_bits = np.array(data_bits).T
        parity_bits = np.array(parity_bits).T

        return data_bits, parity_bits, window_parity
    
    def correction(self, data_bits, recieved_parity_bits, window_parity):
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
        if self.decode_mode:
            data_bits = []

            for index in range(setting.num_of_window_per_channel):
                bs = bit_stream[:, setting.coding_window_size*index:setting.coding_window_size*(index+1)]
                db, rpb, wp = self.decode(bs)
                db = self.correction(db, rpb, wp)
                data_bits.append(db)
            if setting.residual_bits_size:
                data_bits.append(bit_stream[:, -setting.residual_bits_size:])
            data_bits = tf.concat(data_bits, axis=1)
            return data_bits
        else:
            encoded_bit_stream = []
            for index in range(setting.num_of_window_per_channel):
                bs = bit_stream[:, setting.data_bit_size_per_window*index:setting.data_bit_size_per_window*(index+1)]
                ebs, pb = self.encode(bs)
                encoded_bit_stream.append(ebs)
            if setting.residual_bits_size:
                encoded_bit_stream.append(bit_stream[:, -setting.residual_bits_size:])
            encoded_bit_stream = tf.concat(encoded_bit_stream, axis=1)
            encoded_bit_stream = np.array(encoded_bit_stream).astype(int)
            return encoded_bit_stream

class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = y_true * 2 - 1
        return -tf.math.reduce_mean(y_true*y_pred)

class ClipConstraint(tf.keras.constraints.Constraint):
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

class XORMessages(tf.keras.layers.Layer):
    def __init__(self, xor_key):
        super(XORMessages, self).__init__()
        self.xor_key = xor_key

    def call(self, messages):
        return tf.bitwise.bitwise_xor(messages, self.xor_key)

class ImageMessageConcatenation(tf.keras.layers.Layer):
    
    def __init__(self, image_size):
        super(ImageMessageConcatenation, self).__init__()
        self.image_size = image_size
        self.arbitary_message_length = not isinstance(setting.message_bit_per_pixel, int)
        if self.arbitary_message_length:
            self.message_layer = CustomDense(self.image_size*self.image_size*(int(setting.message_bit_per_pixel)+1))

    def call(self, image, messages):
        message = tf.cast(tf.concat(messages, 1), tf.float32)
        message = message * 2 - 1
        if self.arbitary_message_length:
            message = self.message_layer(message)
            message = tf.reshape(message, (-1, self.image_size, self.image_size, int(setting.message_bit_per_pixel)+1))
        else:
            message = tf.reshape(message, (-1, self.image_size, self.image_size, setting.message_bit_per_pixel))
        return tf.concat([image, message], -1)

class ReflectRadding(tf.keras.layers.Layer): # O=[(W−K+P)/S]+1
    def __init__(self, kernel_size):
        super(ReflectRadding, self).__init__()
        pad = kernel_size - 1
        self.upper_pad = pad // 2
        self.lower_pad = pad - self.upper_pad

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.upper_pad, self.lower_pad], [self.upper_pad, self.lower_pad], [0, 0]], 'REFLECT')

class CustomConv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, reflect_padding=False, scale_down_mode=0, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(CustomConv2d, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if reflect_padding:
            if scale_down_mode == 0:
                self.model = [
                    ReflectRadding(kernel_size),
                    tf.keras.layers.Conv2D(num_channel, kernel_size, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 1:
                self.model = [
                    ReflectRadding(kernel_size),
                    tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
            elif scale_down_mode == 2:
                self.model = [
                    ReflectRadding(kernel_size),
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
    
class CustomDense(tf.keras.layers.Layer):

    def __init__(self, output_size, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(CustomDense, self).__init__()

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