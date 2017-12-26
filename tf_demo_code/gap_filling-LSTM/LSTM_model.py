import tensorflow as tf
import numpy as np

# Define LSTM RNN Model
class LSTM_Model():
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, num_layers, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        self.num_layers = num_layers

        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_layers)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32)

            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())

            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data) #(batch, seq_len, rnn_size)

            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)# [(batch, 1, rnn_size)] * seq_len

            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]# [(batch, rnn_size)] * seq_len


        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev, count):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # tf.argmax return max index

            # Get embedded vector
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return (output)

        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # rnn_decoder(
        #     decoder_inputs,       decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        #     initial_state,        initial_state: 2D Tensor with shape [batch_size x cell.state_size]
        #     cell,                 cell: rnn_cell.RNNCell defining the cell function and size.
        #     loop_function=None,
        #     scope=None            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
        # )
        #loop_function: If not None, this function will be applied to the i-th output in order to generate the i+1-st input,
        #  and decoder_inputs will be ignored, except for the first element ("GO" symbol). This can be used for decoding,
        #  but also for training to emulate http://arxiv.org/abs/1506.03099. Signature -- loop_function(prev, i) = next
        #
        #prev is a 2D Tensor of shape [batch_size x output_size],
        #i is an integer, the step number (when advanced control is needed),
        #next is a 2D Tensor of shape [batch_size x input_size].

        # outputs.shape == [batch, rnn_size] * seq_len
        # last_state ==  LSTMStateTuple(c=<shape=(100, 128)>, h=<shape=(100, 128)>)





        # Non inferred outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size]) #[batch * seq_len, rnn_size]

        # Logits and output
        self.logit_output = tf.matmul(output, W) + b #[batch * seq_len, vocab_size]
        self.model_output = tf.nn.softmax(self.logit_output) #[batch * seq_len, vocab_size]


        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])],
                        self.vocab_size)
        # loss.shape ==(5000,) (batch * seq_len)
        # sequence_loss_by_example(
        #     logits,       List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        #     targets,      List of 1D batch-sized int32 Tensors of the same length as logits.
        #     weights,      List of 1D batch-sized float-Tensors of the same length as logits.
        #     average_across_timesteps=True,    If set, divide the returned cost by the total label weight.
        #     softmax_loss_function=None,   Function (labels, logits) -> loss-batch to be used instead of the
        #                                   standard softmax (the default if this is None). Note that to avoid confusion,
        #                                    it is required for the function to accept named arguments.
        #     name=None     Optional name for this operation, default: "sequence_loss_by_example".
        # )
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

    # def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
    def sample(self, sess, words, vocab, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word
        return (out_sentence)