#!/usr/bin/env python
# coding: utf-8

# vgg16_conv_base.py

"""
The script is the self-consistent realization of object-oriented style of the VGG16 model with the 
standand return value of "return model". It is used the static method to replace the construction 
method "def __init__(self,...). Therefore it does not need any parameters including "self". It is 
the much elegant realization of the VGG16 model with the adoption of list data structure. 

In addtion, it has a consolidated structure with the purely Tensorflow 2.x. We set the same 1000 class 
numbers. Please use the dot syntax to call the model if users adopt any client script.

According to the formula of Stanford cs231, W_output = (W-F+2P)/S + 1. W,F,P,S are input width, filter 
width, padding size and stride respectively. It is the apparent result of H_output = W_output since we 
requires the square size of filters.

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

Very Deep Convolutional Networks for Large-Scale Image Recognition
ICLR 2015: https://arxiv.org/abs/1409.1556
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Vgg16(object):
    
    # Adopt the static method to enable the elegant realization of the model  
    @staticmethod 
    def build(input_shape, num_classes):

        # Make the sequential conv_base 
        conv_base = keras.Sequential(
            [    
                keras.Input(shape=input_shape),

                # Conv Block 1 
                layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                # Conv Block 2 
                layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                # Conv Block 3 
                layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                # Conv Block 4 
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                # Conv Block 5 
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            ]
        )

        return conv_base


if __name__ == '__main__':
    
    # Assign the vlaues 
    num_classes = 1000
    input_shape = (227,227,3)
    
    # Use the model to call the function of build() in the Vgg16 class with the dot syntax
    model = Vgg16.build(input_shape, num_classes)

    # Show the Vgg16 Model 
    model.summary()
