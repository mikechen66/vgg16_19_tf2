# vgg16_19_tf2
It is the vgg16 model that adapt to TensorFlow2.0~2.2

The model sets includes commandline, function and  object-oriented style Vgg19 models.
An addition, it divide both"conv_base" or "model" for fintune. It is an attempt to 
re-organize both the conv_base and fc_base for flexible usage.  

In addtion, it has a consolidated structure with the purely Tensorflow 2.x. We set the 
same 1000 class numbers. Please use the following call convention if users adopt any 
client script to call the AlexNet model.
