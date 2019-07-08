import tensorflow as tf


class ConvModel(tf.keras.Model):

    def __init__(self, image_shape=(37, 37), num_classes=2):
        super(ConvModel, self).__init__(name='conv_model_animals')
        self.conv2d_1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(*image_shape, 1))
        self.maxpool2d_1 = tf.keras.layers.MaxPool2D(strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.35)
        self.dense_2 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.3)
        self.dense_3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv2d_1(inputs)
        x = self.maxpool2d_1(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return self.dense_3(x)
