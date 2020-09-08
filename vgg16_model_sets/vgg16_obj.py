#!/usr/bin/env python
# coding: utf-8

# vgg16_obj.py

"""
It is the OOP style VGG16 model without the main function. Please notify that the class
AlexNet has no return value(or None). However, the outer function Vgg16() cna call the 
class (as a function) and then give the summary. 

Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556) (ICLR 2015)

https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D

Keras code: 
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
Tensorflow code:  
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


# Assign the constants 
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
CHANNELS = 3
NUM_CLASSES = 1000


# Define the VGG16 model 
class Vgg16(Sequential):

    def __init__(self, input_shape, num_classes):
        super().__init__()

        # Conv Block 1 
        self.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=input_shape))
        self.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 2 
        self.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 3 
        self.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 4 
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv Block 5 
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # FC Classifer Layer 
        self.add(Flatten())
        self.add(Dense(units=4096, activation="relu"))
        self.add(Dense(units=4096, activation="relu"))
        self.add(Dense(units=num_classes, activation="softmax"))


# Call the AlexNet model 
model = Vgg16((IMAGE_WIDTH,IMAGE_HEIGHT,CHANNELS), NUM_CLASSES)

# show the full model structure of Vgg16 
model.summary()
