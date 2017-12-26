import tensorflow as tf
import numpy as np

return_placeholder = None

def embedding_lookup(input,seq_len,vocab_size, output_size):
    assert input.shape == (input.shape[0],seq_len), \
        "embedding_lookup input shape is {},should be {}".format(inputs.shape, (input.shape[0],seq_len))

    with tf.name_scope('embedding_lookup'):
        embedding_mat = tf.get_variable('embedding_mat', [vocab_size, output_size],
                                        tf.float32, tf.random_normal_initializer())

        output = tf.nn.embedding_lookup(embedding_mat, input)

        assert output.shape == (output.shape[0],seq_len, output_size)
        return output



def bi_lstm(inputs,n_steps,rnn_size,dropout_keep_prob,num_layers):
    # todo: add num_layers
    assert inputs.shape == (inputs.shape[0], n_steps, rnn_size), \
        "bilstm input shape is {},should be {}".format(inputs.shape, (inputs.shape[0], n_steps, rnn_size))

    with tf.name_scope('bi_lstm'):
        cell_unit = tf.contrib.rnn.BasicLSTMCell

        # Forward direction cell
        lstm_forward_cell = cell_unit(rnn_size, forget_bias=1.0)
        lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)

        # Backward direction cell
        lstm_backward_cell = cell_unit(rnn_size, forget_bias=1.0)
        lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=dropout_keep_prob)

        input_embed_split = tf.split(axis=1, num_or_size_splits=n_steps, value=inputs)
        input_embed_split = [tf.squeeze(x, axis=[1]) for x in input_embed_split]


        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                                                            lstm_backward_cell,
                                                                                            input_embed_split,
                                                                                            dtype=tf.float32)

        outputs = tf.transpose(tf.convert_to_tensor(outputs),[1,0,2])# output [batch, seq_len, rnn_hidden_size * 2]
        assert outputs.shape == (outputs.shape[0], n_steps, rnn_size * 2)
        return outputs,(output_state_fw, output_state_bw)

def lstm(inputs,n_steps,rnn_size,dropout_keep_prob,num_layers):
    assert inputs.shape == (inputs.shape[0], n_steps, rnn_size), \
        "lstm input shape is {},should be {}".format(inputs.shape, (inputs.shape[0], n_steps, rnn_size))

    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        # 建立输入的 dropout
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
        return cell

    with tf.name_scope('lstm'):
        if num_layers == 1:
            single_cell = lstm_cell()
        else:
            single_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)

        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=dropout_keep_prob)

        input_embed_split = tf.split(axis=1, num_or_size_splits=n_steps, value=inputs)
        input_embed_split = [tf.squeeze(x, axis=[1]) for x in input_embed_split]

        outputs, states = tf.contrib.rnn.static_rnn(single_cell, input_embed_split, dtype=tf.float32)

        outputs = tf.transpose(tf.convert_to_tensor(outputs), [1, 0, 2])# output [batch, seq_len, rnn_hidden_size * 2]

        assert outputs.shape == (outputs.shape[0], n_steps, rnn_size)
    return outputs,states
def cnn(inputs,filters,kernel_size,poolsize = None,strides = None,data_format='channels_last'):
    # filters = 32,
    # kernel_size = [5, 5],
    # pool_size = [2, 2],
    # strides = [2, 2],
    assert len(inputs.shape) == 4, "cnn input shape must 4,shape:{}".format(inputs.shape)
    with tf.name_scope('cnn'):
        inputs = tf.layers.conv2d(inputs=inputs,
               filters=filters,
               kernel_size=kernel_size,
               strides=[1,1],
               padding='SAME',
               data_format=data_format,
               activity_regularizer=tf.nn.relu)
        if poolsize == None: poolsize = kernel_size
        if strides == None: strides = kernel_size
        out = tf.layers.max_pooling2d(inputs = inputs,
                                      pool_size = poolsize,
                                      strides = strides,
                                      padding='SAME',
                                      data_format=data_format)

    return out

def multi_Desne(input, units, dropout_keep_prob, activations=None):
    if not type(units) == list:
        units = units
    if activations == None or not type(activations) == list:
        activations = [tf.nn.relu for _ in units]

    assert len(units) == len(units)
    with tf.name_scope('multi_Desne'):
        for i in range(len(units)):
            with tf.name_scope('_desne{}'.format(i)):
                u = units[i]
                a = activations[i]

                assert type(u) == int

                input = tf.layers.dense(inputs=input,
                                        units=u,
                                        activity_regularizer=a,
                                        name="dense{}".format(i))
                input = tf.nn.dropout(input,keep_prob=dropout_keep_prob)

    return input

if __name__ == "__main__":
    with tf.Session() as sess:
        loss = 1
        tf.summary.scalar('loss', loss)

        input = tf.placeholder(dtype=tf.int32, shape=[None,10], name='in')
        embedding_lookup(input, 10, 20, 30)

        #
        # input = tf.placeholder(dtype=tf.float32, shape=[None, 2, 3], name='lstm_input')
        # lstm(input, 2, 3, 0.5, 2)
        #
        # #bi_lstm(input, 2, 3, 0.5, 2)
        #
        #
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('./testlog', sess.graph)
        # result = sess.run(merged)
        # writer.add_summary(result, 1)


