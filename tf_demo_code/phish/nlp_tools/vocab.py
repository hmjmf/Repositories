from abc import ABCMeta, abstractmethod

class base_vocab():
    def __init__(self,data_path, vocab_path, extra=["<GO>","<UNK>","<PAD>","<END>"]):
        self.create_by_file(data_path, extra=extra)
        self.save_vocab(vocab_path)
        self.load_voacb_file(vocab_path)

    def create_by_file(self, path, extra=[]):
        self.v = {}
        with open(path) as f:
            for line in f.readlines():
                for word in self.split_line(line):
                    self.v[word] = self.v.get(word,0) + 1
        self.__vocab = list(self.v.keys())
        for i in extra:
            if i not in self.__vocab:
                self.__vocab.append(i)
    def save_vocab(self,path):
        with open(path,'w') as f:
            for i in self.__vocab:
                if not i == "\n":
                    f.write(i + '\n')

    def load_voacb_file(self,path):
        self.w2id = {}
        self.id2w = {}
        with open(path) as f:
            for i,word in enumerate(f.readlines()):
                word = word.replace('\n','')
                if word != "":
                    self.w2id[word] = i
                    self.id2w[i] = word
    def get_id(self,word):
        return self.w2id.get(word,self.w2id.get("<UNK>",-1))
    def get_word(self,id):
        return self.id2w.get(id,"<UNK>")
    @abstractmethod
    def split_line(self,line):
        return line.split(' ')