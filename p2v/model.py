import math
import tensorflow as tf

def c2v(config):
    # set config
    vocab_size = config.get('vocab_size')
    character_size = config['character_size']

    embed_size = config.get('embed_size', 200)
    batch_size = config.get('batch_size', 16)
    neg_sample_size = config.get('neg_sample_size', 100)

    # input
    pos_x = tf.placeholder(tf.int32, [batch_size], name='pos_x') # [batch_size]
    pos_y = tf.placeholder(tf.int32, [batch_size]) # [batch_size x 1]
    neg_y = tf.placeholder(tf.int32, [neg_sample_size], name='neg_y') # [neg_sample_size]

    # embed and softmax weights
    init_width = 0.5 / embed_size
    embed = tf.Variable(tf.random_uniform([character_size, embed_size], -init_width, init_width), name='embed') # [vocab_size x embed_size]
    w = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev=1.0 / math.sqrt(embed_size)), name='w') # [vocab_size x embed_size]
    b = tf.Variable(tf.zeros([vocab_size]), name='b') # [vocab_size]

    global_step = tf.Variable(0, name="global_step")

    # positive examples
    pos_embed = tf.nn.embedding_lookup(embed, pos_x, name='pos_embed') # [batch_size x embed_size]

    pos_w = tf.nn.embedding_lookup(w, pos_y, name='pos_w') # [batch_size x embed_size]
    pos_b = tf.nn.embedding_lookup(b, pos_y, name='pos_b') # [batch_size x 1]

    pos_y_ = tf.reduce_sum(tf.mul(pos_embed, pos_w), 1) + pos_b # [batch_size x 1]

    # negative examples
    neg_w = tf.nn.embedding_lookup(w, neg_y, name='neg_w') # [neg_sample_size x embed_size]
    neg_b = tf.nn.embedding_lookup(b, neg_y, name='neg_b') # [neg_sample_size x 1]
    neg_b = tf.reshape(neg_b, [neg_sample_size])
    neg_y_ = tf.matmul(pos_embed, neg_w, transpose_b=True) + neg_b # [batch_size x neg_sample_size]

    # cross entropy loss
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_y_, tf.ones_like(pos_y_))
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_y_, tf.zeros_like(neg_y_))

    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / batch_size
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # normalized embedding
    nembed = tf.nn.l2_normalize(embed, 1)

    nearby_character = tf.placeholder(dtype=tf.int32)
    nearby_emb = tf.reshape(tf.gather(nembed, nearby_character), [1, embed_size])
    nearby_dist = tf.matmul(nearby_emb, nembed, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(20, character_size))

    return pos_x, pos_y, neg_y, train, loss, nembed, nearby_character, nearby_val, nearby_idx

def w2v(config, pos_x, pos_y):
    # set config
    vocab_size = config.get('vocab_size')
    vocab_counts = config.get('vocab_counts')

    embed_size = config.get('embed_size', 200)
    batch_size = config.get('batch_size', 16)
    neg_sample_size = config.get('neg_sample_size', 100)

    # input
    #pos_x = tf.placeholder(tf.int32, [batch_size], name='pos_x') # [batch_size]
    #pos_y = tf.placeholder(tf.int32, [batch_size]) # [batch_size x 1]
    pos_y_matrix = tf.reshape(tf.cast(pos_y, dtype=tf.int64), [batch_size, 1])

    neg_y, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes = pos_y_matrix,
        num_true = 1,
        num_sampled = neg_sample_size,
        unique = True,
        range_max = vocab_size,
        distortion = 0.75,
        unigrams = vocab_counts.tolist()))

    # embed and softmax weights
    init_width = 0.5 / embed_size
    embed = tf.Variable(tf.random_uniform([vocab_size, embed_size], -init_width, init_width), name='embed') # [vocab_size x embed_size]
    w = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev=1.0 / math.sqrt(embed_size)), name='w') # [vocab_size x embed_size]
    b = tf.Variable(tf.zeros([vocab_size]), name='b') # [vocab_size]

    global_step = tf.Variable(0, name="global_step")

    # positive examples
    pos_embed = tf.nn.embedding_lookup(embed, pos_x, name='pos_embed') # [batch_size x embed_size]

    pos_w = tf.nn.embedding_lookup(w, pos_y, name='pos_w') # [batch_size x embed_size]
    pos_b = tf.nn.embedding_lookup(b, pos_y, name='pos_b') # [batch_size x 1]

    pos_y_ = tf.reduce_sum(tf.mul(pos_embed, pos_w), 1) + pos_b # [batch_size x 1]

    # negative examples
    neg_w = tf.nn.embedding_lookup(w, neg_y, name='neg_w') # [neg_sample_size x embed_size]
    neg_b = tf.nn.embedding_lookup(b, neg_y, name='neg_b') # [neg_sample_size x 1]
    neg_b = tf.reshape(neg_b, [neg_sample_size])
    neg_y_ = tf.matmul(pos_embed, neg_w, transpose_b=True) + neg_b # [batch_size x neg_sample_size]

    # cross entropy loss
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_y_, tf.ones_like(pos_y_))
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_y_, tf.zeros_like(neg_y_))

    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / batch_size
    train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # normalized embedding
    nembed = tf.nn.l2_normalize(embed, 1)

    nearby_word = tf.placeholder(dtype=tf.int32)
    nearby_emb = tf.reshape(tf.gather(nembed, nearby_word), [1, embed_size])
    nearby_dist = tf.matmul(nearby_emb, nembed, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(20, vocab_size))

    return pos_x, pos_y, neg_y, train, loss, nembed, nearby_word, nearby_val, nearby_idx
