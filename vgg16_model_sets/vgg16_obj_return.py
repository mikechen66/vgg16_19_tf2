
#!/usr/bin/env python
# coding: utf-8

# vgg16_obj_return.py

"""
The script is the self-consistent realization of object-oriented programming  of the VGG16 model with 
the explicit "return model" . It is used the static method to replace the construction method. So it 
does not need any the special parameter variable  of self". It is an elegant realization. 

In addtion, it has a consolidated structure with the purely Tensorflow 2.x. We set the same 1000 class 
numbers. Users can use the typical dot syntax to call the VGG16 model. 
model.

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout


class Vgg16(Sequential):

    # Adopt the static method to enbale the elegant realization of the model  
    @staticmethod
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


if __name__ == '__main__':  
    
    # Assign the vlaues 
    input_shape = (227,227,3)
    num_classes = 1000

    # Call the Vgg16  model 
    model = Vgg16().build(input_shape, num_classes)

    # show the full model structure of Vgg16 
    model.summary()
