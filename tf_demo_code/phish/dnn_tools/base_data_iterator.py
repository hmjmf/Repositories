from dnn_tools.data_reader import *
import numpy as np
class base_data_iterator(object):
    def __init__(self,data_reader):
        assert isinstance(data_reader, base_data_reader)

        self.__data_reader = data_reader
        self.__samples, self.__labels = data_reader.get_data()

        assert self.__labels == None or len(self.__samples) == len(self.__labels)


    @abstractmethod
    def next_batch_random(self, batch_size):
        indexs = np.random.choice(range(0,len(self.__samples)), batch_size)
        samples = [self.__samples[i] for i in indexs]

        if self.__labels == None:
            return samples, None
        else:
            labels = [self.__labels[i] for i in indexs]
            return self.process_samples_list(samples),labels

    @abstractmethod
    def nex_batch_with_order(self,batch_size):
        if not hasattr(self, 'star_index'):
            self.star_index = 0

        end_index = min(self.star_index + batch_size,len(self.__samples))
        done = end_index == len(self.__samples)

        samples = self.__samples[self.star_index: end_index]
        if self.__labels == None:
            labels = None
        else:
            labels = self.__labels[self.star_index: end_index]
        self.star_index = end_index

        return self.process_samples_list(samples), labels, done

    @abstractmethod
    def process_samples_list(self,samples_list, config = None):
        return samples_list

    @property
    def samples(self):
        return self.__samples



if __name__ == "__main__":
    '''
        test
    '''

    import os
    try:
        os.mkdir('./test')
    except:
        pass
    with open('./test/data_train', 'w') as f:
        f.write('1,how are you\n0,fuck you\n1,pig is big')
    d = base_data_reader_with_labels('./test/data_train', 'TRAIN')
    di = base_data_iterator(d)
    assert di.nex_batch_with_order(2) == (['how are you\n', 'fuck you\n'], [1, 0], False)
    print(di.next_batch_random(2))

    with open('./test/data_test', 'w') as f:
        f.write('how are you\nfuck you\npig is big')
    d = base_data_reader_with_labels('./test/data_test', 'PREDICT')

    di = base_data_iterator(d)
    assert di.nex_batch_with_order(2) == (['how are you\n', 'fuck you\n'], None, False)
    print(di.next_batch_random(2))




    d = base_data_reader_without_labels('./test/data_test', 'TRAIN')
    di = base_data_iterator(d)

    assert di.nex_batch_with_order(2) == (['how are you\n', 'fuck you\n'], ['how are you\n', 'fuck you\n'], False)
    print(di.next_batch_random(2))


    d = base_data_reader_without_labels('./test/data_test', 'PREDICT')
    di = base_data_iterator(d)
    assert di.nex_batch_with_order(2) == (['how are you\n', 'fuck you\n'], None, False)
    print(di.next_batch_random(2))




