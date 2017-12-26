import tensorflow as tf

# onehot_labels: `[batch_size, num_classes]` target one-hot-encoded labels.
# logits: [batch_size, num_classes] logits outputs of the network .

def softmax_cross_entropy(labels, logits,num_class=None):
    if len(labels.shape) == 1:
        assert not num_class == None
        labels = tf.one_hot(labels, num_class, 1, 0)
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


