import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.misc as sm

def imread(path):
    return sm.imread(path).astype(np.float)

mat_path = "./imagenet-vgg-verydeep-19.mat"
mat = sio.loadmat(mat_path)

for idx, layer in enumerate(mat['layers'][0]):
    try:
        print "[%s]" % idx, ":", layer[0][0][0][0][0].shape
        print "[%sb]" % idx, ":", layer[0][0][0][0][1].shape
    except:
        print "[%s]" % idx, ":", layer

# Conv & bias layer weight examples
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), 
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), 
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes])) 
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

content_img = "./test.png"
style_img = "./style.jpg"

content = imread("./test.png")
style = imread("./style.jpg")

x_shape = (1, ) + content.shape

def conv2d(weight, bias):
    def make_layer(input):
        conv = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding='SAME')
        return tf.nn.bias_add(conv, bias)
    return make_layer

def max_pool():
    def make_layer(input):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return make_layer

for input_image in [content_img, style_img]:
    layer_defs = [
        'conv_1_1', 'relu_1_1','conv1_2', 'relu_1_2', 'pool_1',
        'conv_2_1', 'relu_2_1','conv2_2', 'relu_2_2', 'pool_2',

        'conv_3_1', 'relu_3_1','conv3_2', 'relu_3_2',
        'conv_3_3', 'relu_3_3', 'conv3_4', 'relu_3_4', 'pool_3',
        'conv_4_1', 'relu_4_1','conv4_2', 'relu_4_2',
        'conv_4_3', 'relu_4_3', 'conv4_4', 'relu_4_4', 'pool_4',

        'conv_5_1', 'relu_5_1','conv4_2', 'relu_5_2',
        'conv_5_3', 'relu_5_3', 'conv4_4', 'relu_5_4'
    ]

    mean = mat['normalization'][0][0][0] # (224, 224, 3)
    mean_pixel = np.mean(mean, axis=(0,1))

    # mat['layers'] : [1 x 43]
    constants = mat['layers'][0]

    layers = []
    for idx, layer_def in enumerate(layer_defs):
        if 'conv' in layer_def:
            w = constants[idx][0][0][0][0][0]
            w = np.transpose(w, (1, 0, 2, 3)) # [3 x 3 x 3 x 64] : 3x3 conv, 3 input, 64 output channels
            b = constants[idx][0][0][0][0][1] # [1 x 64] -> [64]
            b = b.reshape(-1)                 # [64]
            ops = conv2d(w, b)
        elif 'relu' in layer_def:
            ops = tf.nn.relu
        elif 'pool' in layer_def:
            ops = max_pool()
        else:
            raise "Wrong layer def : %s" % layer_def

        if not layers:
            layers.append(ops(input_image))
        else:
            layers.append(ops(layers[-1]))

    layer_dict = dict(zip(layer_defs, layers)

def preprocess(image, mean_pixel):
    return image - mean_pixel

def unprocess(image, mean_pixel):
    return image + mean_pixel
