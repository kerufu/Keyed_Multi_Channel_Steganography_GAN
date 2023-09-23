import tensorflow as tf

import setting

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

class image_label_concatenation(tf.keras.layers.Layer):
    
    def __init__(self, image_size, image_channel):
        super(image_label_concatenation, self).__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.label_layer = custom_dense(self.image_size*self.image_size*self.image_channel)

    def call(self, image, label):
        label = self.label_layer(label)
        label = tf.reshape(label, (-1, self.image_size, self.image_size, self.image_channel))
        return tf.concat([image, label], -1)

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, scale_down=False, maxpooling=False, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_conv2d, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if scale_down:
            if maxpooling:
                self.model = [
                    tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                    tf.keras.layers.MaxPool2D(),
                ]
            else:
                self.model = [
                    tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
                ]
        else:
            self.model = [
                tf.keras.layers.Conv2D(num_channel, kernel_size, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
            ]

        self.model += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation)
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
    
class custom_conv2dtp(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, scale_up=False, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_conv2dtp, self).__init__()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if scale_up:
            self.model = [
                tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
            ]
        else:
            self.model = [
                tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=1, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2(), kernel_constraint=kernel_constraint),
            ]

        self.model += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
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
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation)
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
    