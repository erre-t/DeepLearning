import numpy as np
from matplotlib import pyplot as plt
import random


class RNN():
    def __init__(self):

        self.m = 5
        self.eta = 0.1
        self.sig = 0.01
        self.seq_length = 25
        self.eps = 1e-8


        self.book_data = [char for char in open("goblet_book.txt").read()]
        self.book_chars = list(set(self.book_data))
        self.K = len(self.book_chars)

        self.U = np.random.randn(self.m, self.K) * self.sig
        self.W = np.random.randn(self.m, self.m) * self.sig
        self.V = np.random.randn(self.K, self.m) * self.sig
        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.K,1))

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.grad_b = np.zeros((self.K, self.seq_length))
        self.grad_c = np.zeros((self.K, self.seq_length))
        self.grad_a = np.zeros((self.m, self.seq_length))
        self.grad_h = np.zeros((self.m, self.seq_length))
        self.grad_o = np.zeros((self.K, self.seq_length))

        self.best_U = []
        self.best_W = []
        self.best_V = []
        self.best_b = []
        self.best_c = []

    def soft_max(self, s):
        """ Standard definition of the soft_max function """
        a = np.exp(s) / np.sum(np.exp(s), axis = 0)
        return a

    def char_to_ind(self, char):
        one_hot = np.zeros((self.K))
        one_hot[self.book_chars.index(char)] = 1
        return one_hot

    def ind_to_char(self, one_hot):
        char = self.book_chars[np.where(one_hot == 1)[0][0]]
        return char

    def forward_pass(self, X, Y, h0):

        a = np.zeros((self.m, self.seq_length))
        o = np.zeros((self.K, self.seq_length))
        p = np.zeros((self.K, self.seq_length))
        h = np.zeros((self.m, self.seq_length))
        h[:,0] = h0[:,self.seq_length-1].reshape(self.m)
        #h = h0
        for t in range(self.seq_length):
            x = X[:,t].reshape((self.K, 1))

            a[:,t] = (np.dot(self.W, h[:,t-1].reshape(self.m,1)) + np.dot(self.U, x) + self.b).reshape(self.m)
            h[:,t] = (np.tanh(a[:,t])).reshape(self.m)
            o[:,t] = (np.dot(self.V, h[:,t].reshape(self.m,1)) + self.c).reshape(self.K)
            p[:,t] = (self.soft_max(o[:,t])).reshape(self.K)

        loss = self.compute_loss(Y, p)

        return loss, a, h, p

    def backward_pass(self, X, Y, a, h, p):

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))
        self.grad_b = np.zeros((self.K, self.seq_length))
        self.grad_c = np.zeros((self.K, self.seq_length))
        self.grad_a = np.zeros((self.m, self.seq_length))
        self.grad_h = np.zeros((self.m, self.seq_length))
        self.grad_o = np.zeros((self.K, self.seq_length))

        self.grad_o = -(Y - p)

        self.grad_V = np.dot(self.grad_o, np.transpose(h))

        self.grad_h[:,-1] = np.dot(self.grad_o[:,-1], self.V)
        self.grad_a[:,-1] = np.dot(self.grad_h[:,-1], np.diag(1-np.tanh(a[:,-1])**2))

        #for t in range(self.seq_length-2, -1, -1):
        for t in reversed(range(self.seq_length-1)):
            self.grad_h[:,t] = np.dot(self.grad_o[:,t], self.V) + np.dot(self.grad_a[:,t+1], self.W)
            self.grad_a[:,t] = np.dot(self.grad_h[:,t], np.diag(1-np.tanh(a[:,t])**2))

        self.grad_c = self.grad_o.sum(axis = 1).reshape(self.K, 1) # ????
        self.grad_b = self.grad_a.sum(axis = 1).reshape(self.m, 1) # ????

        for t in range(1,self.seq_length):
            self.grad_W += np.dot(self.grad_a[:,t].reshape(self.m, 1), np.transpose(h[:,t-1].reshape(self.m,1))) # ????


        self.grad_U = np.dot(self.grad_a, np.transpose(X))


        self.grad_U = np.maximum(np.minimum(self.grad_U, 5), -5)
        self.grad_W = np.maximum(np.minimum(self.grad_W, 5), -5)
        self.grad_V = np.maximum(np.minimum(self.grad_V, 5), -5)
        self.grad_b = np.maximum(np.minimum(self.grad_b, 5), -5)
        self.grad_c = np.maximum(np.minimum(self.grad_c, 5), -5)

    def minmax(self):

        self.grad_b[self.grad_b > 5] = 5
        self.grad_b[self.grad_b < -5] = -5
        self.grad_c[self.grad_c > 5] = 5
        self.grad_c[self.grad_c < -5] = -5
        self.grad_V[self.grad_V > 5] = 5
        self.grad_V[self.grad_V < -5] = -5
        self.grad_W[self.grad_W > 5] = 5
        self.grad_W[self.grad_W < -5] = -5
        self.grad_U[self.grad_U > 5] = 5
        self.grad_U[self.grad_U < -5] = -5

    def compute_loss(self, Y, p):
        loss = 0
        for t in range(self.seq_length):
            y = Y[:,t].reshape(self.K,1)
            loss -= np.log(np.dot(np.transpose(y), p[:,t].reshape(self.K,1)))
        return loss[0][0]

    def get_one_hot_sequence(self, e):

        X_chars = self.book_data[e: e + self.seq_length]
        Y_chars = self.book_data[e+1: e + self.seq_length+1]

        X = np.zeros((self.K, self.seq_length))
        Y = np.zeros((self.K, self.seq_length))

        for c in range(self.seq_length):

            X[:,c] = self.char_to_ind(X_chars[c]) # Måste vara (80,)

        for c in range(self.seq_length):

            Y[:,c] = self.char_to_ind(Y_chars[c])


        return X, Y

    def compute_grads_num(self, X, Y, h):

        hh = np.zeros((self.m, self.seq_length))

        grad_num_b = np.zeros(self.grad_b.shape)
        grad_num_c = np.zeros(self.grad_c.shape)
        grad_num_U = np.zeros(self.grad_U.shape)
        grad_num_W = np.zeros(self.grad_W.shape)
        grad_num_V = np.zeros(self.grad_V.shape)

        for i in range(self.b.shape[0]):
            self.b[i,0] -= h
            l1, _, _, p1 = self.forward_pass(X, Y, hh)
            self.b[i,0] += 2*h
            l2, _, _, p2 = self.forward_pass(X, Y, hh)
            grad_num_b[i] = (l2-l1) / (2*h)
            self.b[i,0] -= h

        print("grad_num_b finished ...")
        hh = np.zeros((self.m, self.seq_length))

        for i in range(self.c.shape[0]):

            self.c[i,0] -= h
            l1, _, _, p1 = self.forward_pass(X, Y, hh)
            self.c[i,0] += 2*h
            l2, _, _, p2 = self.forward_pass(X, Y, hh)
            grad_num_c[i] = (l2-l1) / (2*h)
            self.c[i,0] -= h

        print("grad_num_c finished ...")
        hh = np.zeros((self.m, self.seq_length))


        for i in range(self.U.shape[0]):
          for j in range(self.U.shape[1]):
            self.U[i][j] -= h
            l1,_, _, p1 = self.forward_pass(X, Y, hh)
            self.U[i][j] += 2*h
            l2, _, _, p2 = self.forward_pass(X, Y, hh)
            grad_num_U[i][j] = (l2-l1) / (2*h)
            self.U[i][j] -= h


        print("grad_num_U finished ...")
        hh = np.zeros((self.m, self.seq_length))

        for i in range(self.W.shape[0]):
          for j in range(self.W.shape[1]):
            self.W[i][j] -= h
            l1, _, _, p1 = self.forward_pass(X, Y, hh)
            self.W[i][j] += 2*h
            l2, _, _, p2 = self.forward_pass(X, Y, hh)
            grad_num_W[i][j] = (l2-l1) / (2*h)
            self.W[i][j] -= h


        print("grad_num_W finished ...")
        hh = np.zeros((self.m, self.seq_length))

        for i in range(self.V.shape[0]):
          for j in range(self.V.shape[1]):
            self.V[i][j] -= h
            l1, _, _, p1 = self.forward_pass(X, Y, hh)
            self.V[i][j] += 2*h
            l2, _, _, p2 = self.forward_pass(X, Y, hh)
            grad_num_V[i][j] = (l2-l1) / (2*h)
            self.V[i][j] -= h

        print("grad_num_V finished ...")


        return grad_num_b, grad_num_c, grad_num_U, grad_num_V, grad_num_W

    def check_gradients(self, X, Y):
        hh = np.zeros((self.m, self.seq_length))
        [loss, a, h, p] = self.forward_pass(X, Y, hh)
        self.backward_pass(X, Y, a, h, p)
        [grad_num_b, grad_num_c, grad_num_U, grad_num_V, grad_num_W] = self.compute_grads_num(X, Y, 1e-4)


        ('SUMS: Numerically     Analytically')
        print(str(np.sum(grad_num_b)) + "   " + str(np.sum(self.grad_b)))
        print(str(np.sum(grad_num_c)) + "   " + str(np.sum(self.grad_c)))
        print(str(np.sum(grad_num_U)) + "   " + str(np.sum(self.grad_U)))
        print(str(np.sum(grad_num_W)) + "   " + str(np.sum(self.grad_W)))
        print(str(np.sum(grad_num_V)) + "   " + str(np.sum(self.grad_V)))


        diff_b = np.sum(abs(grad_num_b - self.grad_b)) / max(self.eps, (np.sum(abs(grad_num_b)) + np.sum(abs(self.grad_b))))
        diff_c = np.sum(abs(grad_num_c - self.grad_c)) / max(self.eps, (np.sum(abs(grad_num_c)) + np.sum(abs(self.grad_c))))

        diff_U = np.sum(abs(grad_num_U - self.grad_U)) / max(self.eps, (np.sum(abs(grad_num_U)) + np.sum(abs(self.grad_U))))
        diff_W = np.sum(abs(grad_num_W - self.grad_W)) / max(self.eps, (np.sum(abs(grad_num_W)) + np.sum(abs(self.grad_W))))
        diff_V = np.sum(abs(grad_num_V - self.grad_V)) / max(self.eps, (np.sum(abs(grad_num_V)) + np.sum(abs(self.grad_V))))

        print("------- Relative error  -------")
        print('diff_b ' + str(diff_b))
        print('diff_c ' + str(diff_c))

        print('diff_U ' + str(diff_U))
        print('diff_W ' + str(diff_W))
        print('diff_V ' + str(diff_V))

    def train(self):
        e = 0
        update_steps = 0
        m_b = np.zeros(self.b.shape)
        m_c = np.zeros(self.c.shape)
        m_U = np.zeros(self.U.shape)
        m_W = np.zeros(self.W.shape)
        m_V = np.zeros(self.V.shape)
        h0 = np.zeros((self.m, self.seq_length))

        losses = []
        smooth_losses = []
        smooth_loss = -np.log(1 / self.K) * self.seq_length

        epochs = 5
        for epoch in range(epochs):

            while update_steps < 1000:    #while e < (len(self.book_data)-self.seq_length-1):

                X, Y = self.get_one_hot_sequence(e)
                loss, a, h, p = self.forward_pass(X, Y, h0)
                h0 = h
                self.backward_pass(X, Y, a, h0, p)
                #self.minmax()
                self.ada_grad(m_b, m_c, m_U, m_W, m_V)

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                if (update_steps%100 == 0):
                #    Y = self.synthesize(X[:,0], h)
                #    self.print_sequence(Y)
                    losses.append(loss)
                    smooth_losses.append(smooth_loss)


                e += self.seq_length
                update_steps += 1

            e = 0
            update_steps = 0
            h0 = np.zeros((self.m, self.seq_length))
            print("Epoch " + str(epoch + 1) + " done...")

        optimal_model = np.argmin(smooth_losses)

        return optimal_model, losses, smooth_losses

    def ada_grad(self, m_b, m_c, m_U, m_W, m_V):

        m_b += self.grad_b**2
        self.b += -(self.eta * self.grad_b) / np.sqrt(self.eps + m_b)
        self.best_b.append(self.b)

        m_c += self.grad_c**2
        self.c += -(self.eta * self.grad_c) / np.sqrt(self.eps + m_c)
        self.best_c.append(self.c)

        m_U += self.grad_U**2
        self.U += -(self.eta * self.grad_U) / np.sqrt(self.eps + m_U)
        self.best_U.append(self.U)

        m_V += self.grad_V**2
        self.V += -(self.eta * self.grad_V) / np.sqrt(self.eps + m_V)
        self.best_V.append(self.V)

        m_W += self.grad_W**2
        self.W += -(self.eta * self.grad_W) / np.sqrt(self.eps + m_W)
        self.best_W.append(self.W)

    def synthesize(self, x0, h0):
        self.seq_length = 200

        #x = "a"
        #x = self.char_to_ind(x0).reshape(self.K, 1)
        x = x0.reshape(self.K, 1)

        Y = np.zeros((self.K, self.seq_length))
        a = np.zeros((self.m, self.seq_length))
        o = np.zeros((self.K, self.seq_length))
        p = np.zeros((self.K, self.seq_length))
        h = np.zeros((self.m, self.seq_length))

        #self.hprev = np.zeros((self.m))
        h[:,0] = h0[:,25-1].reshape(self.m)


        Y = np.zeros((self.K, self.seq_length))
        for t in range(self.seq_length):

            a[:,t] = (np.dot(self.W, h[:,t-1].reshape(self.m,1)) + np.dot(self.U, x) + self.b).reshape(self.m)
            h[:,t] = (np.tanh(a[:,t])).reshape(self.m)
            o[:,t] = (np.dot(self.V, h[:,t].reshape(self.m,1)) + self.c).reshape(self.K)
            p[:,t] = (self.soft_max(o[:,t])).reshape(self.K)

            cp = np.cumsum(p[:,t])
            dist = np.random.uniform(low = 0, high = 1)
            i = np.nonzero(cp - dist > 0)
            ii = i[0][0]
            xnext = self.book_chars[ii]
            Y[:,t] = self.char_to_ind(xnext)
            x = self.char_to_ind(xnext).reshape(self.K,1)

        self.seq_length = 25


        return Y


    def print_sequence(self, X):
        sequence = []
        for t in range(X.shape[1]):
            sequence.append(self.ind_to_char(X[:,t]))
        print("Sequence:  " + ''.join(sequence))



def plotLoss(losses, smooth_losses): # Update steps
    t = np.arange(0, len(losses), 1)
    fig, ax = plt.subplots()
    ax.plot(t, losses, label = 'Loss')
    ax.plot(t, smooth_losses, label = 'Smooth loss')
    ax.legend()
    ax.plot(t, losses, t, smooth_losses)
    ax.set(xlabel='Update steps', ylabel='Loss')
    plt.show()

def main():
    np.random.seed(10)

    rnn = RNN()
    #[X, Y] = rnn.get_one_hot_sequence(0)
    #rnn.check_gradients(X, Y)

    ## Check gradients
    [X, Y] = rnn.get_one_hot_sequence(0)
    rnn.check_gradients(X, Y)

    ## Training
    #ptimal_model, losses, smooth_losses = rnn.train()

    ## Synthesize
    #Y = rnn.synthesize(X[:,0])
    #rnn.print_sequence(Y)
    # Plots

    #plotLoss(losses, smooth_losses)




main()
