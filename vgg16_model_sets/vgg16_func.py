#!/usr/bin/env python
# coding: utf-8

# vgg16_func.py

"""
It is a typical functional programming for the classical vgg16 model. Please notify 
that it has the explicit return. 

Very Deep Convolutional Networks for Large-Scale Image Recognition
ICLR 2015: https://arxiv.org/abs/1409.1556

Keras code: 
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py

Tensorflow code:  
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def vgg16(input_shape, num_classes):

    model = Sequential()

    # Conv Block 1 
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Conv Block 2 
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Conv Block 3 
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Conv Block 4 
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Conv Block 5 
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # FC Classifer Layer 
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))
 
    return model 

# Provide the constants for the function. 
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
CHANNELS = 3
NUM_CLASSES = 1000


# Call the AlexNet model 
model = vgg16((IMAGE_WIDTH,IMAGE_HEIGHT,CHANNELS), NUM_CLASSES)

# show the full model structure of Vgg16 
model.summary()
