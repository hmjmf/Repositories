import tensorflow as tf
import numpy as np

from dnn_tools.base_net import *
from dnn_tools.base_data_iterator import *
from dnn_tools.score import *
import time

class estimator():
    def __init__(self,sess ,mode, net, data_iter, eval_data_iter,model_dir):
        assert mode == 'TRAIN' or mode == 'PREDICT' , 'UNKOW MODE'
        assert isinstance(net, base_net)
        assert isinstance(data_iter, base_data_iterator)
        assert isinstance(eval_data_iter, base_data_iterator)

        self.sess = sess
        self.mode = mode
        self.net = net
        self.data_iter = data_iter
        self.eval_data_iter = eval_data_iter
        self.model = self.create_or_load_model(model_dir)


    def create_or_load_model(self, model_dir):
        print("creating model")
        self.net.create_net()
        print("creating model over")
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if self.mode == 'TRAIN':
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                self.net.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print("sess.run(tf.global_variables_initializer())")
                self.sess.run(tf.global_variables_initializer())
            return


        if self.mode == 'PREDICT':
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                self.net.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise FileNotFoundError







    def train(self, max_step, report_step, eval_stpe, patience,batch_size,eval_batch_size, leatn_rate_decay_factor):

        train_loss_average = 0
        start_time = time.time()
        last_report_time = start_time
        beat_eval_loss = 100000
        max_patience = patience

        for current_step in range(max_step):

            samples, labels = self.data_iter.next_batch_random(batch_size)
            assert not labels == None
            assert len(samples) == len(labels)

            train_loss_step = self.net.run_train_step(samples, labels)
            train_loss_step = self.process_loss(train_loss_step)

            train_loss_average += train_loss_step

            if current_step % report_step == 0:
                speed = (batch_size * report_step) // (time.time() - last_report_time)
                last_report_time = time.time()
                all_time = time.time() - start_time
                train_loss_average = train_loss_average / report_step
                print("[{},step:{},speed:{}/s]train_loss:{}".format(all_time, current_step, speed, train_loss_average))
                train_loss_average = 0


            if current_step % eval_stpe == 0:
                all_time = time.time()-start_time

                samples, labels = self.data_iter.next_batch_random(eval_batch_size)
                eval_loss, predictedY = self.net.run_eval_step(samples, labels)

                print("\n-----EVAL BEGIN-----")
                print("[{},step:{}]eval_loss:{},patience:{}".format(all_time,current_step,eval_loss,patience))

                get_score(labels, predictedY)
                print("-----EVAL end-----\n")

                if eval_loss > beat_eval_loss:
                    print('patience:{}'.format(patience))
                    patience -= 1
                    self.net.run_leatn_rate_decay(leatn_rate_decay_factor)
                else:
                    beat_eval_loss = eval_loss
                    patience = max_patience

                    #todo: add saver

                if patience == 0:
                    print("train over because can't do better")
                    break

        print("train over")

    def eval(self,eval_data_iter,eval_batch_size):
        done = False
        predictedY_sum = []
        labels_sum = []
        loss_sum = 0
        while not done:
            samples, labels, done = eval_data_iter.nex_batch_with_order(eval_batch_size)
            loss, predictedY = self.net.run_eval_step(samples, labels)
            loss_sum += self.process_loss(loss)
            labels_sum.extend(labels)
            predictedY_sum.extend(predictedY)

        print("[eval]loss:{}".format(loss_sum / len(labels_sum)))
        get_score(labels_sum, predictedY_sum)


    def predicted(self,predicted_data_iter, batch_size):
        while not done:
            samples, _, done = eval_data_iter.nex_batch_with_order(batch_size)
            predictedY = self.net.run_predicted_step(samples)
            predictedY_sum.extend(predictedY)
        return predictedY_sum


    @abstractmethod
    def process_loss(self, loss):
        return loss



