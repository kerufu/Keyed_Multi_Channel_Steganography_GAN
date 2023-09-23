import tensorflow as tf

import setting

class ClipConstraint(tf.keras.constraints.Constraint):
    
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

class xor_messages(tf.keras.layers.Layer):

    def __init__(self, xor_key):
        self.xor_key = xor_key

    def call(self, messages):
        return tf.bitwise.bitwise_xor(messages, self.xor_key)

class image_label_concatenation(tf.keras.layers.Layer):
    
    def __init__(self, image_size, image_channel):
        super(image_label_concatenation, self).__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.label_layer = tf.keras.layers.Dense(self.image_size*self.image_size*self.image_channel)

    def call(self, image, label):
        label = self.label_layer(label)
        label = tf.reshape(label, (-1, self.image_size, self.image_size, self.image_channel))
        return tf.concat([image, label], -1)

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, maxpooling=False, regularize_kernal=True, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_conv2d, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if maxpooling:
            self.model = [
                tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
                tf.keras.layers.MaxPool2D(),
            ]
        else:
            self.model = [
                tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
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

    def __init__(self, num_channel, kernel_size, regularize_kernal=True, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_conv2dtp, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.model = [
            tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
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

    def __init__(self, output_size, regularize_kernal=True, clip_kernal=False, dropout=False, activation="leaky_relu"):
        super(custom_dense, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.model = [
            tf.keras.layers.Dense(output_size, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
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
    
class generator(tf.keras.Model):
    def __init__(self, key):
        super(generator, self).__init__()
        self.encoder = [
            custom_conv2d(64, 3),
            custom_conv2d(128, 5),
            custom_conv2d(256, 3),
            custom_conv2d(512, 3),
        ]
        self.xor_layer = xor_messages(key)
        self.concat_layer = image_label_concatenation(setting.image_size//16, 512)
        self.decoder = [
            custom_conv2dtp(256, 3),
            custom_conv2dtp(128, 3),
            custom_conv2dtp(64, 5),
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
            custom_conv2d(64, 3),
            custom_conv2d(128, 5),
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
            custom_conv2d(64, 3),
            custom_conv2d(128, 5),
            custom_conv2d(256, 3),
            custom_conv2d(512, 3),
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
