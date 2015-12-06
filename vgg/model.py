import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.misc as sm

sess = tf.InteractiveSession()

def imread(path):
    return sm.imread(path).astype(np.float)

mat_path = "./imagenet-vgg-verydeep-19.mat"
mat = sio.loadmat(mat_path)

def build_conv2d(weight, bias):
    def make_layer(input):
        return tf.nn.conv2d(input, weight, [1,2,2,1], )
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
x = tf.placeholder(tf.int32, [None, width, height, channel_size])

for idx, tag in enumerate(layer_tags):
    if 'conv' in tag:
        op = build_weight(prev)
    elif 'relu' in tag:
        op = tf.nn.relu(prev)
    elif 'pool' in tag:
        op = tf.nn.max_pool(prev), 
    else:
        raise("Error: %s not found" % tag)

    if prev:
        layers.append(op(prev))
    else:
        layers.append(op(x))
