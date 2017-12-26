from dnn_tools.base_net import base_net
from dnn_tools.models import *
from dnn_tools.loss import *
from dnn_tools.train_op import *
import tensorflow as tf
import numpy as np

class phish_net(base_net):
    def __init__(self, sess, vocab_size, rnn_size, seq_len, dropout_keep_prob,num_layers,learn_rate,num_class):
        self.sess = sess
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_class = num_class
        self.dropout_keep_prob = tf.Variable(float(dropout_keep_prob), trainable=False, dtype=tf.float32)
        self.dropout_open_op = self.dropout_keep_prob.assign(dropout_keep_prob)
        self.dropout_close_op = self.dropout_keep_prob.assign(1.0)
        self.learn_rate = tf.Variable(float(learn_rate), trainable=False, dtype=tf.float32)

        self.saver_ = tf.train.Saver()

    def create_net(self):
        self.input_plhd = tf.placeholder(tf.int32,shape=[None, self.seq_len],name='input')
        self.traget_plhd = tf.placeholder(tf.int32, shape=[None, self.num_class], name='output')


        embedding_out = embedding_lookup(self.input_plhd, self.seq_len, self.vocab_size, self.rnn_size)
        #(batch , seq_len, rnn_size)

        _, (bi_lstm_output_state_fw, bi_lstm_output_state_bw) = \
            bi_lstm(embedding_out, self.seq_len, self.rnn_size, self.dropout_keep_prob, self.num_layers)



        c,h = bi_lstm_output_state_bw
        bi_lstm_out = tf.concat([c, h], 1)



        feature_len = np.cumprod([int(i) for i in bi_lstm_out.shape[1:]])[-1]



        logits = multi_Desne(input=bi_lstm_out,
                             units=[int(feature_len // 4),
                                    int(feature_len // 16),
                                    self.num_class],
                             dropout_keep_prob=self.dropout_keep_prob)

        self.logits = logits

        self.loss = softmax_cross_entropy(self.traget_plhd, self.logits ,num_class=self.num_class)

        self.train_op=min_loss(self.loss,tf.train.AdamOptimizer(self.learn_rate),max_gradient_norm=None)


    def run_train_step(self, samples, labels):
        self.sess.run(self.dropout_open_op)


        input_feed = {self.input_plhd : samples,
                      self.traget_plhd : self.sess.run(tf.one_hot(labels, self.num_class, 1, 0))}

        out_feed = [self.loss,self.train_op]

        loss, _ = self.sess.run(fetches=out_feed, feed_dict = input_feed)

        return loss



    def run_eval_step(self, samples, labels):
        self.sess.run(self.dropout_close_op)

        input_feed = {self.input_plhd: samples,
                      self.traget_plhd: self.sess.run(tf.one_hot(labels, self.num_class, 1, 0))}

        out_feed = [self.loss, tf.argmax(self.logits,1)]

        loss, logits= self.sess.run(out_feed, input_feed)



        return loss, logits



    def run_predicted_step(self, samples):
        self.sess.run(self.dropout_close_op)

        input_feed = {self.input_plhd: samples}

        out_feed = tf.argmax(self.logits,1)

        logits = self.sess.run(out_feed, input_feed)

        return logits



    def run_leatn_rate_decay(self, factor):
        learn_rate_decay_op = self.learn_rate.assign(self.learn_rate * factor)
        self.sess.run(learn_rate_decay_op)



    @property
    def saver(self):
        return self.saver_