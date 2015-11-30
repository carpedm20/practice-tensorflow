import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.misc as sm

def imread(path):
    return sm.imread(path).astype(np.float)

image_shape = [];

mat_path = "./imagenet-vgg-verydeep-19.mat"
data = sio.loadmat(mat_path)

content_img = "./test.png"
style_img = "./style.jpg"

content = imread("./test.png")
style = imread("./style.jpg")

x_shape = (1, ) + content.shape
r = tf.placeholder([x_shape])

def conv2d(input):
    def make_layer(weight, bias):
        return tf.nn.conv2d(input, weight, strid=[], padding='SAME')

    return make_layer(weight, bias)

def model_factory():
    x = tf.placeholder()
    y = tf.placeholder()

    layers = ['conv1_1', 'relu_1_1','conv1_2', 'relu_1_2']

    for layer in layers:
        make_layer = conv2d(x)
        layer = make_layer(weight, bias)


