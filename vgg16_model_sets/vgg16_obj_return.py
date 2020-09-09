#!/usr/bin/env python
# coding: utf-8

# alexnet_obj_return.py

"""
The script is the self-consistent realization of object-oriented style of the AlexNet model with the 
standand return value "return model". It is used the static method to replace the construction method
"def __init__(self,...). Therefore it does not need any parameters including "self". It is the much
elegant realization of the VGG16 model. 

In addtion, it has a consolidated structure with the purely Tensorflow 2.x. We set the same 1000 class 
numbers. Please use the following call convention if users adopt any client script to call the AlexNet 
model.

## 1. Delete the model, summary and arguments within the script of the AlexNet model.

## 2. Add the constants in the client script.
# IMAGE_WIDTH = 227
# IMAGE_HEIGHT = 227
# CHANNELS = 3
# NUM_CLASSES = 1000
## 3.Assign the vlaues in the client script.
# INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
##4. Use a client script to call the AlexNet model 
# model = vgg16_obj_return.Vgg16.build(INPUT_SHAPE, NUM_CLASSES)

According to the formula of Stanford cs231, W_output = (W-F+2P)/S + 1. W,F,P,S are input width, filter 
width, padding size and stride respectively. It is the apparent result of H_output = W_output since we 
requires the square size of filters.

Stanford c231n 
https://cs231n.github.io/convolutional-networks/#conv

AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Michael Guerzhoy and Davi Frossard, 2017, AlexNet implementation in TensorFlow, with weights Details: 
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


class Vgg16(Sequential):

    # Adopt the static method to enbale the elegant realization of the model  
    @staticmethod
    # -def build(width, height, depth, classes, reg=0.0002):
    def build(input_shape, num_classes):

        model = Sequential()

        # Conv Block 1 
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal',
                        input_shape=input_shape))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 2 
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 3 
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 4 
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 5 
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer='he_normal'))
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

# Assign the vlaues 
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)


# Use the model to call the function of build() in the AlexNet class with the dot syntax
model = Vgg16.build(INPUT_SHAPE, NUM_CLASSES)

# Show the AlexNet Model 
model.summary()