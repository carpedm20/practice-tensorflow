import tensorflow as tf

def p2v(vocab_size, embed_size):
    pos_x = tf.placeholder('float', [batch_size, 1])
    pos_y = tf.placeholder('float', [batch_size, 1])
    neg_y = tf.placeholder('float', [batch_size * neg_sample_size, 1])

    width = 0.5 / embed_size
    embed = tf.Variable(tf.random_uniform([vocab_size, embed_size], -width, +width))
    w = tf.Variable(tf.random_uniform([vocab_size, embed_size], -width, +width))
    b = tf.Variable(tf.random_uniform([vocab_size], -width, +width))

    # positive examples
    pos_embed = tf.nn.embedding_lookup(embed, pos_x)

    pos_w = tf.nn.embedding_lookup(w, pos_y)
    pos_b = tf.nn.embedding_lookup(w, pos_y)

    pos_y_ = pos_embed * pos_w + pos_b

    # negative examples
    neg_w = tf.nn.embedding_lookup(embed, neg_y)
    neg_b = tf.nn.embedding_lookup(embed, neg_y)

    pos_y_ = tf.matmul(pos_embed, neg_y, transpose_b=True) + pos_b

    pos_loss = tf.nn.softmax_cross_entropy_with_logits(pos_y_, tf.ones_like(pos_y_))
    neg_loss = tf.nn.softmax_cross_entropy_with_logits(neg_y_, tf.zeros_like(neg_y_))

    loss = pos_loss + neg_loss

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return pos_x, pos_y, neg_y, train
