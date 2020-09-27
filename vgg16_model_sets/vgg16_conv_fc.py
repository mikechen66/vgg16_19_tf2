#!/usr/bin/env python
# coding: utf-8

# alexnet_obj_return.py

"""
The script is the self-consistent realization of object-oriented style of the Vgg16 model with the 
return value "conv_base" or "model". It is an attempt to re-organize both the conv_base and fc_base
for flexible usage.  

In addtion, it has a consolidated structure with the purely Tensorflow 2.x. We set the same 1000 class 
numbers. Please use the following call convention if users adopt any client script to call the AlexNet 
model.

Issues:

The script includes deplicated line of code because the function of Concatenate or k.concatenate can 
not assembe the abstact layers in the Sequential Model. Therefore, we need to eliminatee the repeated 
code for elegant realization. 

# https://keras.io/api/layers/merging_layers/concatenate/
# -from tensorflow.keras.layers import Concatenate
or
# -import keras.backend as K
or 
# -model = merge(conv_base + fc_base) 

According to the formula of Stanford cs231, W_output = (W-F+2P)/S + 1. W,F,P,S are input width, filter 
width, padding size and stride respectively. It is the apparent result of H_output = W_output since we 
requires the square size of filters.

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

Very Deep Convolutional Networks for Large-Scale Image Recognition
ICLR 2015: https://arxiv.org/abs/1409.1556

Keras code: 
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py

Tensorflow code:  
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Vgg16(object):
    # Adopt the static method to enable the elegant realization of the model  
    @staticmethod 
    def build(input_shape, num_classes, exclude_fc=None):

        if exclude_fc: 
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

        else: 
            # Make the sequential conv_base 
            model = keras.Sequential(
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

                    # FC classifier 
                    layers.Flatten(),
                    layers.Dense(units=4096, activation="relu"),
                    layers.Dense(units=4096, activation="relu"),
                    layers.Dense(units=num_classes, activation="softmax")

                ]
            )

            return model
        
        
if __name__ == '__main__':
    
    # Assign the vlaues 
    input_shape = (227,227,3)
    num_classes = 1000
    # -exclude_fc = False 
    exclude_fc = True

    # Call the function of build() in the Vgg16 class with the dot syntax
    # -model = Vgg16().build(input_shape, num_classes, exclude_fc)
    conv_base = Vgg16().build(input_shape, num_classes, exclude_fc)

    # Show the Vgg16 Model 
    # -model.summary()
    conv_base.summary()
