import numpy as np
from matplotlib import pyplot as plt


class RNN():
    def __init__(self):
        self.m = 5
        self.eta = 0.1
        self.sig = 0.01
        self.seq_length = 25

        self.book_data = [char for char in open("goblet_book.txt").read()]
        self.book_chars = list(set(self.book_data))
        self.K = len(self.book_chars)

        self.U = np.random.rand(self.m, self.K) * self.sig
        self.W = np.random.rand(self.m, self.m) * self.sig
        self.V = np.random.rand(self.K, self.m) * self.sig
        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K,1))

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.grad_b = np.zeros((self.K, self.seq_length))
        self.grad_c = np.zeros((self.K, self.seq_length))

        self.h_prev = np.zeros((self.m,1))


    def soft_max(self, s):
        """ Standard definition of the soft_max function """
        return np.exp(s) / np.sum(np.exp(s), axis=0)

    def char_to_ind(self, char):
        one_hot = np.zeros((self.K))
        one_hot[self.book_chars.index(char)] = 1
        return one_hot

    def ind_to_char(self, one_hot):
        char = self.book_chars[np.where(one_hot == 1)[0][0]]
        return char

    def forward_pass(self, X):

        a = np.zeros((self.m, self.seq_length))
        h = np.zeros((self.m, self.seq_length))
        o = np.zeros((self.K, self.seq_length))
        p = np.zeros((self.K, self.seq_length))


        #h[:,0] = np.zeros((self.m))

        x0 = X[:,0].reshape((self.K, 1))

        """
        a[:,0] = (np.dot(self.W, h[:,0].reshape(self.m,1)) + np.dot(self.U, x0) + self.b).reshape(self.m)
        h[:,0] = (np.tanh(a[:,0])).reshape(self.m)
        o[:,0] = (np.dot(self.V, h[:,0].reshape(self.m,1)) + self.c).reshape(self.K)
        p[:,0] = (self.soft_max(o[:,0])).reshape(self.K)
        """

        for t in range(1, self.seq_length):
            x = X[:,t].reshape((self.K, 1))

            a[:,t] = (np.dot(self.W, self.h_prev.reshape(self.m,1)) + np.dot(self.U, x) + self.b).reshape(self.m)
            h[:,t] = (np.tanh(a[:,t])).reshape(self.m)
            o[:,t] = (np.dot(self.V, h[:,t].reshape(self.m,1)) + self.c).reshape(self.K)
            p[:,t] = (self.soft_max(o[:,t])).reshape(self.K)

            #a[t] = a_curr.reshape(self.m) #reshape from (m,1) to (m,)
            #h[t] = h_curr.reshape(self.m) #reshape from (m,1) to (m,)
            #p[t] = p_curr.reshape(self.K) #reshape from (k,1) to (k,)
            self.h_prev = h[:,t]

        #self.h_prev = h_curr
        #print(h.shape)
        #a = a.reshape(self.m, self.seq_length)
        #h = h.reshape(self.m, self.seq_length)
        #p = p.reshape(self.K, self.seq_length)
        #print(a.shape)

        return a, h, p

    def backward_pass(self, X, Y, a, h, p):

        self.U = np.random.rand(self.m, self.K) * self.sig
        self.W = np.random.rand(self.m, self.m) * self.sig
        self.V = np.random.rand(self.K, self.m) * self.sig
        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K,1))

        grad_a = np.zeros((self.m, self.seq_length))
        grad_h = np.zeros((self.m, self.seq_length))
        grad_o = np.zeros((self.K, self.seq_length))

        for t in range(self.seq_length):
            y_t = Y[:,t].reshape(self.K,1)
            p_t = p[:,t].reshape(self.K,1)
            grad_o[:,t] = (-(y_t - p_t)).reshape(self.K)

        for t in range(self.seq_length):
            self.grad_V += np.dot(grad_o[:,t].reshape(self.K, 1), np.transpose(h[:,t].reshape(self.m,1)))

        grad_h[:,-1] = np.dot(grad_o[:,-1], self.V)
        grad_a[:,-1] = np.dot(grad_h[:,-1], np.diag(1-np.tanh(a[:,-1])**2))


        for t in range(self.seq_length-2, -1, -1):
            grad_h[:,t] = np.dot(grad_o[:,t], self.V) + np.dot(grad_a[:,t+1], self.W)
            grad_a[:,t] = np.dot(grad_h[:,t], np.diag(1-np.tanh(a[:,t])**2))

        self.grad_c = grad_o.sum(axis = 1).reshape(self.K, 1) # ????
        self.grad_b = grad_a.sum(axis = 1).reshape(self.m, 1) # ????

        for t in range(self.seq_length):
            self.grad_W += np.outer(grad_a[:,t].reshape(self.m, 1), h[:,t-1]) # ????


        for t in range(self.seq_length):
            x = X[:,t].reshape(self.K, 1)
            self.grad_U += np.dot(grad_a[:,t].reshape(self.m, 1), np.transpose(x))

        #return self.grad_b, self.grad_c, self.grad_U, self.grad_V, self.grad_W

    def compute_loss(self, Y, p):
        loss = 0
        for t in range(self.seq_length):
            y = Y[:,t].reshape(self.K,1)
            loss -= np.log(np.dot(np.transpose(y), p[:,t].reshape(self.K,1)))

        return loss

    def synthesize(self):

        x = "a"
        x = self.char_to_ind(x)
        h = np.zeros((self.m, 1))

        indices = []
        Y = np.zeros((self.K, self.seq_length))

        for t in range(0,self.seq_length):
            a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
            h = np.tanh(a)
            o = np.dot(self.V, h) + self.c
            p = self.soft_max(o)

            i = np.random.randint(0, self.K)

            """
            cp = np.cumsum(p)
            a = np.random.rand()
            i = [(ind, val) for ind, val in enumerate(cp) if (val-a) > 0]
            i = i[0][0]
            """

            indices.append(i)

            xnext = self.book_chars[i]
            x = self.char_to_ind(xnext)
            Y[:,t] = x

        return Y

    def get_one_hot_sequence(self):
        X_chars = self.book_data[0:self.seq_length]
        Y_chars = self.book_data[1:self.seq_length+1]

        X = np.zeros((self.K, self.seq_length))
        Y = np.zeros((self.K, self.seq_length))

        for c in range(self.seq_length):

            X[:,c] = self.char_to_ind(X_chars[c])

        for c in range(self.seq_length):

            Y[:,c] = self.char_to_ind(Y_chars[c])


        return X, Y

    def compute_grads_num(self, X, Y, h):

        grad_num_b = np.zeros(self.grad_b.shape)
        grad_num_c = np.zeros(self.grad_c.shape)
        grad_num_U = np.zeros(self.grad_U.shape)
        grad_num_W = np.zeros(self.grad_W.shape)
        grad_num_V = np.zeros(self.grad_V.shape)

        for i in range(self.b.shape[0]):
            self.b[i,0] -= h
            _, _, p1 = self.forward_pass(X)
            l1 = self.compute_loss(Y, p1)
            self.b[i,0] += 2*h
            _, _, p2 = self.forward_pass(X)
            l2 = self.compute_loss(Y, p2)
            grad_num_b[i] = (l2-l1) / (2*h)
            self.b[i,0] -= h

        print("grad_num_b finished ...")

        for i in range(self.c.shape[0]):
            self.c[i,0] -= h
            _, _, p1 = self.forward_pass(X)
            l1 = self.compute_loss(Y, p1)
            self.c[i,0] += 2*h
            _, _, p2 = self.forward_pass(X)
            l2 = self.compute_loss(Y, p2)
            grad_num_c[i] = (l2-l1) / (2*h)
            self.c[i,0] -= h

        print("grad_num_c finished ...")


        for i in range(self.U.shape[0]):
          for j in range(self.U.shape[1]):
            self.U[i][j] -= h
            _, _, p1 = self.forward_pass(X)
            l1 = self.compute_loss(Y, p1)
            self.U[i][j] += 2*h
            _, _, p2 = self.forward_pass(X)
            l2 = self.compute_loss(Y, p2)
            grad_num_U[i][j] = (l2-l1) / (2*h)
            self.U[i][j] -= h


        print("grad_num_U finished ...")

        for i in range(self.W.shape[0]):
          for j in range(self.W.shape[1]):
            self.W[i][j] -= h
            _, _, p1 = self.forward_pass(X)
            l1 = self.compute_loss(Y, p1)
            self.W[i][j] += 2*h
            _, _, p2 = self.forward_pass(X)
            l2 = self.compute_loss(Y, p2)
            grad_num_W[i][j] = (l2-l1) / (2*h)
            self.W[i][j] -= h


        print("grad_num_W finished ...")

        for i in range(self.V.shape[0]):
          for j in range(self.V.shape[1]):
            self.V[i][j] -= h
            _, _, p1 = self.forward_pass(X)
            l1 = self.compute_loss(Y, p1)
            self.V[i][j] += 2*h
            _, _, p2 = self.forward_pass(X)
            l2 = self.compute_loss(Y, p2)
            grad_num_V[i][j] = (l2-l1) / (2*h)
            self.V[i][j] -= h

        print("grad_num_V finished ...")


        return grad_num_b, grad_num_c, grad_num_U, grad_num_V, grad_num_W

def main():


    rnn = RNN()
    X, Y = rnn.get_one_hot_sequence()
    #Y = rnn.synthesize()
    a, h, p = rnn.forward_pass(X)
    rnn.backward_pass(X, Y, a, h, p)
    [grad_num_b, grad_num_c, grad_num_U, grad_num_V, grad_num_W] = rnn.compute_grads_num(X, Y, 1e-4)


    print('SUMS: Numerically     Analytically')
    print(str(np.sum(grad_num_b)) + "   " + str(np.sum(rnn.grad_b)))
    print(str(np.sum(grad_num_c)) + "   " + str(np.sum(rnn.grad_c)))
    print(str(np.sum(grad_num_U)) + "   " + str(np.sum(rnn.grad_U)))
    print(str(np.sum(grad_num_W)) + "   " + str(np.sum(rnn.grad_W)))
    print(str(np.sum(grad_num_V)) + "   " + str(np.sum(rnn.grad_V)))






    eps = 0.000000000001
    diff_b = np.sum(abs(grad_num_b - rnn.grad_b)) / max(eps, (np.sum(abs(grad_num_b)) + np.sum(abs(rnn.grad_b))))
    diff_c = np.sum(abs(grad_num_c - rnn.grad_c)) / max(eps, (np.sum(abs(grad_num_c)) + np.sum(abs(rnn.grad_c))))

    diff_U = np.sum(abs(grad_num_U - rnn.grad_U)) / max(eps, (np.sum(abs(grad_num_U)) + np.sum(abs(rnn.grad_U))))
    diff_W = np.sum(abs(grad_num_W - rnn.grad_W)) / max(eps, (np.sum(abs(grad_num_W)) + np.sum(abs(rnn.grad_W))))
    diff_V = np.sum(abs(grad_num_V - rnn.grad_V)) / max(eps, (np.sum(abs(grad_num_V)) + np.sum(abs(rnn.grad_V))))

    print("------- Relative error  -------")
    print('diff_b  ' + str(diff_b))
    print('diff_c ' + str(diff_c))

    print('diff_U ' + str(diff_U))
    print('diff_W ' + str(diff_W))
    print('diff_V ' + str(diff_V))


    """
    sequence = []
    for i in range(rnn.seq_length):
        sequence.append(rnn.ind_to_char(Y[:,i]))
    print(''.join(sequence))
    """




main()
