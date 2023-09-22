import tensorflow as tf

import setting

class ClipConstraint(tf.keras.constraints.Constraint):
    
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

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
    def __init__(self):
        super(generator, self).__init__()
        self.encoder = [
            custom_conv2d(64, 3),
            custom_conv2d(128, 5),
            custom_conv2d(256, 3),
            custom_conv2d(512, 3),
            tf.keras.layers.Flatten(),
            custom_dense(setting.feature_size, activation="tanh"),
        ]
        self.decoder = [
            custom_dense(setting.image_size*setting.image_size*2),  
            tf.keras.layers.Reshape((setting.image_size//16, setting.image_size//16, 512)),
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
        m = tf.cast(tf.concat(messages, 1), tf.float32)
        image = tf.concat([m, image], 1)
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
            tf.keras.layers.Dense(setting.message_size, activation="tanh")
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
