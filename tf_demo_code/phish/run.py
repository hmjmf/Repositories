from dnn_tools.base_data_iterator import *
from dnn_tools.data_reader import *
from dnn_tools.estimator import *
from nlp_tools.vocab import *
from phish_net import *


seq_len_cut_prob = 0.8
model_dir = "./model/"
max_step = 500
report_step = 50
eval_stpe = 1000
patience = 10
batch_size = 64
eval_batch_size = 512
leatn_rate_decay_factor = 0.5
rnn_size = 128
dropout_keep_prob = 0.5
num_layers = 2
learn_rate = 0.1
num_class = 2
seq_len = 69

class vocab(base_vocab):
    def split_line(self,line):
        return [i for i in line[2:]]

v = vocab("../dataset/phish/train.txt","../dataset/phish/vocab.txt")

class data_reader_with_labels(base_data_reader_with_labels):
    def process_sample(self, sample):
        return [v.get_id(i) for i in sample]



class data_iterator(base_data_iterator):
    def __get_seq_len(self, cut_prob):
        s_l = {}
        for i in self.samples:
            l = len(i)
            s_l[l] = s_l.get(l, 0) + 1
        c = 0
        for i in range(max(list(s_l.keys()))):
            c += s_l.get(i, 0)
            if c >= len(self.samples) * cut_prob:
                return i
    def process_samples_list(self,samples_list,config = None):
        if not hasattr(self, 'seq_len'):
            #self.seq_len = self.__get_seq_len(seq_len_cut_prob)
            self.seq_len = seq_len
        s = []
        for i in samples_list:
            if len(i) <= self.seq_len:
                s_ = [v.get_id("<PAD>")] * (self.seq_len - len(i)) + i

            else:
                s_ = i[:self.seq_len]
            assert len(s_) == self.seq_len
            s.append(s_)
        return s


data_reader_train_ = data_reader_with_labels(data_path="../dataset/phish/train.txt",mode='TRAIN')
data_iter_train_ = data_iterator(data_reader_train_)

data_reader_eval_ = data_reader_with_labels(data_path="../dataset/phish/eval.txt",mode='TRAIN')
data_iter_eval_ = data_iterator(data_reader_eval_)



config = tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

vocab_size = len(list(v.w2id.keys()))
seq_len = 69



phish_net_ = phish_net(sess, vocab_size, rnn_size, seq_len, dropout_keep_prob,num_layers,learn_rate,num_class)



e = estimator(sess,mode="TRAIN",net=phish_net_, data_iter=data_iter_train_, eval_data_iter=data_iter_eval_, model_dir=model_dir)



all_vars = tf.global_variables()
for var in all_vars:
    print(var.name)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./testlog', sess.graph)
result = sess.run(merged)
writer.add_summary(result, 1)


e.train(max_step, report_step, eval_stpe, patience,batch_size,eval_batch_size, leatn_rate_decay_factor)

e.eval(eval_data_iter=data_iter_eval_, eval_batch_size=128)


data_reader_test_ = data_reader_with_labels(data_path="../dataset/phish/test.txt",mode='TRAIN')
data_iter_test_ = data_iterator(data_reader_test_)
e.eval(eval_data_iter=data_iter_test_, eval_batch_size=128)

