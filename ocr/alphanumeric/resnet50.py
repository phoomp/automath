import tensorflow as tf
from tensorflow import keras

import tensorflow.keras.layers


class ResNet50(tf.keras.Model):
    def __init__(self, n_classes):
        super(ResNet50, self).__init__()
        self.input = layers.Input
        self.resnet_block = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=,
            input_shape=None,
            pooling=None,
            classes=n_classes,
        )
    
        self.top = layers.Dense(n_classes)
        