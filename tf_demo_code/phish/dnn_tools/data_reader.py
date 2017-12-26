import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class base_data_reader(object):
    @abstractmethod
    def load_data_with_label(self, data_path):
        raise NotImplementedError

    @abstractmethod
    def load_data_without_label(self, data_path):
        raise NotImplementedError

    @abstractmethod
    def get_data(self):
        raise NotImplementedError

class base_data_reader_with_labels(base_data_reader):
    '''
        data file line should like:
            1,how are you
            0,fuck you
            no title
    '''

    def __init__(self,data_path,mode):
        assert mode == 'TRAIN' or mode == 'PREDICT', 'UNKOW MODE'
        self.mode = mode
        if mode == 'TRAIN':
            self.samples, self.labels = self.load_data_with_label(data_path)
        else:
            self.samples = self.load_data_without_label(data_path)

    def load_data_with_label(self,data_path):
        labels = []
        samples = []
        with open(data_path,'r') as f:
            for i in f.readlines():
                try:
                    label = self.process_label(i[0])
                    sample = self.process_sample(i[2:])
                except:
                    print('load data error:{}'.format(i))
                else:
                    labels.append(label)
                    samples.append(sample)

        assert len(labels) == len(samples)
        print("get data with label {}".format(len(labels)))
        print('labels like:{}'.format(labels[:2]))
        print('samples like:{}'.format(samples[:2]))
        return samples, labels
    def load_data_without_label(self,data_path):
        with open(data_path, 'r') as f:
            samples = [self.process_sample(i) for i in f.readlines()]
        print("get data without label {}".format(len(samples)))
        print('samples like:{}'.format(samples[:2]))
        return samples

    def get_data(self):
        if self.mode == 'TRAIN':
            return self.samples, self.labels
        else:
            return self.samples, None

    @abstractmethod
    def process_label(self,label):
        return int(label)

    @abstractmethod
    def process_sample(self, sample):
        return sample



class base_data_reader_without_labels(base_data_reader):
    '''
        data file line should like:
            how are you
            fuck you
            no title
    '''

    def __init__(self,data_path,mode):
        assert mode == 'TRAIN' or mode == 'PREDICT', 'UNKOW MODE'
        self.mode = mode
        if mode == 'TRAIN':
            self.samples, self.labels = self.load_data_with_label(data_path)
        else:
            self.samples = self.load_data_without_label(data_path)

    def load_data_with_label(self,data_path):
        labels = []
        samples = []
        with open(data_path,'r') as f:
            for i in f.readlines():
                try:
                    label = self.process_label(i)
                    sample = self.process_sample(i)
                except:
                    print('load data error:{}'.format(i))
                else:
                    labels.append(label)
                    samples.append(sample)

        assert len(labels) == len(samples)
        print("get data with label {}".format(len(labels)))
        print('labels like:{}'.format(labels[:2]))
        print('samples like:{}'.format(samples[:2]))
        return samples, labels

    def load_data_without_label(self,data_path):
        with open(data_path, 'r') as f:
            samples = [self.process_sample(i) for i in f.readlines()]
        print("get data without label {}".format(len(samples)))
        print('samples like:{}'.format(samples[:2]))
        return samples

    @abstractmethod
    def process_label(self, label):
        return label

    @abstractmethod
    def process_sample(self, sample):
        return sample

    def get_data(self):
        if self.mode == 'TRAIN':
            return self.samples, self.labels
        else:
            return self.samples, None

if __name__ == "__main__":
    '''
        test
    '''

    import os
    try:
        os.mkdir('./test')
    except:
        pass
    with open('./test/data_train','w') as f:
        f.write('1,how are you\n0,fuck you\n1,pig is big')
    d = base_data_reader_with_labels('./test/data_train','TRAIN')
    assert d.labels[0] == 1
    assert d.samples[0] == "how are you\n"

    with open('./test/data_test','w') as f:
        f.write('how are you\nfuck you\npig is big')
    d = base_data_reader_with_labels('./test/data_test', 'PREDICT')
    assert d.samples[0] == "how are you\n"

    d = base_data_reader_without_labels('./test/data_test', 'TRAIN')
    assert d.labels[0] == "how are you\n"
    assert d.samples[0] == "how are you\n"

    d = base_data_reader_without_labels('./test/data_test', 'PREDICT')
    assert d.samples[0] == "how are you\n"

