import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

sess = tf.InteractiveSession()

#def build_model(config):
config = {}

batch_size = config.get('batch_size', 20)

input_size = config.get('input_size', 1000) # set of index in dictionary
hidden_size = config.get('hidden_size', 200)
seq_size = config.get('seq_size', 50)
layer_size = config.get('layer_size', 2)

cell = rnn_cell.BasicLSTMCell(hidden_size)
x = tf.placeholder(tf.int32, [None, seq_size])
y = tf.placeholder(tf.int32, [None, seq_size])
initial_state = cell.zero_state(batch_size, tf.float32)

#with tf.variable_scope('rnn'):
w = tf.get_variable('softmax_w', [hidden_size, input_size])
b = tf.get_variable('softmax_b', [input_size])
#with tf.device('/cpu:0'):

# [input_size x hidden_size]
embed = tf.get_variable('embed', [input_size, hidden_size])

# [batch_size x seq_size x hidden_size]
input_set = tf.nn.embedding_lookup(embed, x)
# [batch_size x 1 x hidden_size] x seq_size
inputs = tf.split(1, seq_size, input_set)
# [batch_size x hidden_size] x seq_size
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

def loop(prev, _):
    prev = tf.nn.xw_plus_b(prev, w, b)
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    # [batch_size x hidden_size]
    return tf.nn.embedding_lookup(embed, prev_symbol)

infer = False

# outputs : [batch_size x hidden_size]
# states : [batch_size x state_size]
outputs, states = seq2seq.rnn_decoder(inputs, initial_state, cell, loop_function=loop if infer else None, scope='rnn')
output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
