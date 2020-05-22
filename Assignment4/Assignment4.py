import numpy as np
from matplotlib import pyplot as plt


class RNN():
    def __init__(self):
        self.m = 100
        self.eta = 0.1
        self.sig = 0.01
        self.seq_length = 25

        self.book_data = [char for char in open("goblet_book.txt").read()]
        self.book_chars = list(set(self.book_data))
        self.K = len(self.book_chars)

        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K,1))


        self.U = np.random.rand(self.m, self.K) * self.sig
        self.W = np.random.rand(self.m, self.m) * self.sig
        self.V = np.random.rand(self.K, self.m) * self.sig

def SoftMax(s):
    """ Standard definition of the softmax function """
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def one_hot_matrix(K, n):
    Y_hot = np.zeros((len(K), n))
    Y_hot[np.arange(len(K)), n] = 1
    Y_hot = np.transpose(Y_hot)

    return Y_hot

def char_to_ind(book_chars, char):
    one_hot = [0] * len(book_chars)
    one_hot[book_chars.index(char)] = 1

    return one_hot

#def ind_to_char(book_chars, one_hot):
#    chars =  [i for i in cp if (i-a) > 0]

#    return


def synthesize(RNN, h0, x0, n):

    indices = []
    #a = np.zeros((RNN.m, RNN.seq_length))
    #h = np.zeros((RNN.m, RNN.seq_length))
    #o = np.zeros((RNN.K, RNN.seq_length))
    #p = np.zeros((RNN.K, RNN.seq_length))

    a = []
    h = []
    o = []
    p = []

    x = []
    x.append(x0)
    h.append(h0)

    a.append(np.dot(RNN.W, h0) + np.dot(RNN.U, x0) + RNN.b)
    h.append(np.tanh(a[0]))
    o.append(np.dot(RNN.V, h0) + RNN.c)
    p.append(SoftMax(o[0]))

    cp = list(np.cumsum(p[0]))
    a = np.random.rand()
    val, ind =  [ x for ind, val in enumerate(list(cp)) if (val-a) > 0]
    indices.append(ind)

    xnext = RNN.book_chars[ii]
    x.append(xnext)

    for t in range(1,n):
        a.append(np.dot(RNN.W, h[t-1]) + np.dot(RNN.U, x[t]) + RNN.b)
        h.append(np.tanh(a[t]))
        o.append(np.dot(RNN.V, h[t]) + RNN.c)
        p.append(SoftMax(o[t]))

        cp = np.cumsum(p[t])
        a = np.random.rand()
        ixs =  [i for i in cp if (i-a) > 0]
        ii = ixs(1)
        indices.append(ii)

        xnext = RNN.book_chars[ii]
        x.append(xnext)

    Y = one_hot_matrix(RNN.book_chars, indices)

    return Y



def main():


    rnn = RNN()

    x0 = "a"
    x0 = char_to_ind(rnn.book_chars, x0)

    h0 = np.zeros((rnn.m, 1))



    synthesize(rnn, h0, x0, rnn.seq_length)


main()
