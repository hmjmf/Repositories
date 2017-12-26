def accuracy(label, predictedY):
    assert len(label) == len(predictedY)
    return sum([label[i] == predictedY[i] for i in range(len(label))]) / len(label)

def f1_score(label, predictedY):
    assert len(label) == len(predictedY)
    try:
        p = len([x for x in range(len(label)) if label[x] == predictedY[x] == 1]) * 1.0 / len([x for x in label if x == 1])
        r = len([x for x in range(len(label)) if label[x] == predictedY[x] == 1]) / len([x for x in predictedY if x == 1])

        return p * r * 200 / (p + r), p, r
    except ZeroDivisionError:
        return -1,0,0

def get_score(label, predictedY):
    print("===== score =====")
    print("accuracy:{}".format(accuracy(label,predictedY)))
    f1,precision,recall = f1_score(label, predictedY)
    print("f1:{},precision:{},recall:{}".format(f1,precision,recall))

