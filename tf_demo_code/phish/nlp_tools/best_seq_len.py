
def get_seq_len(cut_prob):
    s_l = {}
    for i in self.__samples:
        l = len(i)
        s_l[l] = s_l.get(l,0) + 1
    c = 0
    for i in range(max(list(s_l.keys()))):
        c += s_l.get(i,0)
        if c >= len(self.__samples) * cut_prob:
            return i