import os
import random
import pickle
import numpy as np
import tensorflow as tf

from model import w2v
from data import build_vocab
from embedding import gen_word2vec as word2vec

sess = tf.InteractiveSession()

# config
config = {
    'batch_size'      : 16,
    'embed_size'      : 200,
    'neg_sample_size' : 100,
}

# data
data, unique_neg_data, idx2word, word2idx, vocab = build_vocab()

name2idx = dict([(name, idx) for idx, name in enumerate(data.keys())])
idx2name = dict([(idx, name) for idx, name in enumerate(data.keys())])

vocab_size = len(idx2word)
character_size = len(data)

# model
(words, counts, words_per_epoch, epoch, words, pos_x, pos_y) = word2vec.skipgram(
    filename = 'text8',
    batch_size = 16,
    window_size = 2,
    min_count = 5,
    subsample = 1e-3
)

(vocab_words, vocab_counts, words_per_epoch) = sess.run([words, counts, words_per_epoch])

config['vocab_size'] = 71290
config['vocab_counts'] = vocab_counts
pos_x, pos_y, neg_y, train, loss, nembed, nearby_word, nearby_val, nearby_idx = w2v(config, pos_x, pos_y)

# train
step_size = 100000
learning_rate = 0.025
window_size = 5
min_count = 5
subsample = 1e-3

batch_idx = 0
data_idx = {}

for i in xrange(len(data)):
    data_idx[i] = 0

def generate_batch(config, data, unique_neg_data):
    global batch_idx, data_idx

    batch_size = config['batch_size']
    neg_sample_size = config['neg_sample_size']

    batch = data.values()[batch_idx]
    neg_batch = unique_neg_data.values()[batch_idx]
    idx = data_idx[batch_idx]

    data_pos_x = np.ones(batch_size) * batch_idx
    data_pos_y = np.ndarray(shape=batch_size, dtype=np.int32)
    data_neg_y = np.ndarray(shape=neg_sample_size, dtype=np.int32)

    for i in xrange(batch_size):
        data_pos_y[i] = batch[idx]
        idx = (idx + 1) % len(batch)

    for i, neg_y_idx in enumerate(random.sample(set(neg_batch), neg_sample_size)):
        data_neg_y[i] = neg_y_idx

    data_idx[batch_idx] = idx
    batch_idx = (batch_idx + 1) % len(data)

    return data_pos_x, data_pos_y, data_neg_y

batch_pkl = "./batch.pkl"

if os.path.isfile(batch_pkl):
    batch = pickle.load(open(batch_pkl))
else:
    batch = []
    for i in xrange(step_size):
        batch.append(generate_batch(config, data, unique_neg_data))

    pickle.dump(batch, open(batch_pkl, "wb"))

#with tf.Session() as sess:
if True:
    sess.run(tf.initialize_all_variables())
    average_loss = 0

    batch_size = config['batch_size']
    neg_sample_size = config['neg_sample_size']

    for step in xrange(step_size):
        data_pos_x, data_pos_y, data_neg_y = batch[step]
        feed_dict = {pos_x: data_pos_x, pos_y: np.reshape(data_pos_y, (batch_size)), neg_y: np.reshape(data_neg_y, (neg_sample_size))}
        _, loss_val = sess.run([train, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            print("Average loss at step ", step, ": ", average_loss)
            acharacter_sizeverage_loss = 0

def nearby(words, num=20):
    ids = np.array([name2idx.get(x, 0) for x in words])
    vals, idx = sess.run(
        [nearby_val, nearby_idx], {nearby_word: ids})
    for i in xrange(len(words)):
        print(words[i])
        print()
        for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
            print("%-20s %6.4f" % (idx2name[neighbor], distance))

def nearby_with_idx(idxs, num=20):
    ids = np.array(idxs)
    vals, idx = sess.run(
        [nearby_val, nearby_idx], {nearby_word: ids})
    for i in xrange(len(idxs)):
        print(idx2name[idxs[i]])
        print()
        for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
            print("%-20s %6.4f" % (idx2name[neighbor], distance))
