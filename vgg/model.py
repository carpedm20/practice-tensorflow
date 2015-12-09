import os
import numpy as np
import scipy.io as sio
import scipy.misc as sm
import tensorflow as tf

from tensorflow.python.platform import gfile

from utils import *

sess = tf.InteractiveSession()

def imread(path):
    return sm.imread(path).astype(np.float)

mat_path = "./imagenet-vgg-verydeep-19.mat"
mat = sio.loadmat(mat_path)

def build_conv2d(w, b):
    def make_layer(input):
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
        return tf.nn.bias_add(conv, np.reshape(b, (b.shape[1])))
    return make_layer

def build_pool():
    def make_layer(input):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return make_layer

layer_tags = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv4_2', 'relu5_2',
    'conv5_3', 'relu5_3', 'conv4_4', 'relu5_4'
]

mean = mat['normalization'][0][0][0] # (224, 224, 3)
mean_pixel = np.mean(mean, axis=(0,1))

# mat['layers'] : [1 x 43]
constants = mat['layers'][0]

prev = None
x = tf.placeholder(tf.float32, [None, None, None, None])

layers = []
for idx, tag in enumerate(layer_tags):
    if 'conv' in tag:
        op = build_conv2d(constants[idx][0][0][0][0][0], constants[idx][0][0][0][0][1])
    elif 'relu' in tag:
        op = tf.nn.relu
    elif 'pool' in tag:
        op = build_pool()
    else:
        raise("Error: %s not found" % tag)

    if layers:
        layers.append(op(layers[-1]))
    else:
        layers.append(op(x))

image_path = os.path.join('./', 'test.png')
if not os.path.isfile(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
image_data = imread(image_path)

sess = tf.InteractiveSession()
sess.run(layers[-1], feed_dict={x: [image_data]})
