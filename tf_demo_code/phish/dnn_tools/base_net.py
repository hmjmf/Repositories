import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class base_net():

    def __init__(self,sess):
        self.sess = sess

    @abstractmethod
    def create_net(self):
        raise NotImplementedError

    @abstractmethod
    def run_train_step(self, samples, labels):
        raise NotImplementedError

    @abstractmethod
    def run_eval_step(self, samples, labels):
        # rember drop out
        raise NotImplementedError

    @abstractmethod
    def run_predicted_step(self, samples):
        raise NotImplementedError

    @abstractmethod
    def run_leatn_rate_decay(self, factor):

        raise NotImplementedError

    @abstractmethod
    def predict(self,input):
        raise NotImplementedError

    @abstractmethod
    def save_net(self):
        raise NotImplementedError

    @property
    def saver(self):
        return None

