import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class RNN():
    def __init__(self):

        [self.trainX, self.trainY, self.validationX, self.validationY, self.testX, self.testY] = getData()
        #[self.trainX, self.trainY, self.validationX, self.validationY, self.testX, self.testY] = getDataOriginal()

        self.d = self.trainX.shape[0] # Number of dimensions
        self.n = self.trainX.shape[1] # Number of images
        self.K = self.trainY.shape[0] # Number of classes
        self.m = [self.d, 50, 30, 20, 20, 10, 10, 10, 10, self.K] # nodes in the hidden layer
        self.k = len(self.m)-1 # Number of layers

        self.alpha = 0.9
        self.gamma = []
        self.beta = []
        self.W = []
        self.b = []
        for i in range(1, self.k+1):
            self.W.append(np.random.normal(0, np.sqrt(2 / self.m[i-1]), (self.m[i],self.m[i-1])))
            self.b.append(np.zeros((self.m[i],1)))

        self.cycles = 2
        self.batch_size = 100
        self.n_s = int(5 * (self.n / self.batch_size))
        self.epochs = int((2 * self.n_s  * self.cycles * self.batch_size)/ self.n)
        self.GDparams = {'n_batch': self.batch_size, 'eta_min': 1e-5, 'eta_max': 1e-1, 'n_s': self.n_s, 'n_epochs': self.epochs}
        self.lamb = 0.005

    def check_gradients(self):

        [gn_W, gn_b] = self.ComputeGradients(self.trainX[:,0:5], self.trainY[:,0:5])
        [ga_W, ga_b] = self.ComputeGradsNum(self.trainX[:,0:5], self.trainY[:,0:5], 1e-5)

        print('SUMS:')
        diff_W = []
        diff_b = []
        eps = 0.000000000001

        #print(str(len(ga_W)) + "  " + str(len(gn_W)))

        for l in range(0, self.k):
            print(" ---- Layer " + str(l) + " ---")
            print("ga_W: " + str(np.sum(ga_W[l])) + " ga_b: " + str(np.sum(ga_b[l])))
            print("gn_W: " + str(np.sum(gn_W[len(self.W)-1-l])) + " gn_b: " + str(np.sum(gn_b[len(self.W)-1-l])))

            print(str(ga_W[l].shape) + "   " + str(gn_W[len(self.W)-1-l].shape))

            diff_W = np.sum(abs(ga_W[l] - gn_W[len(self.W)-1-l])) / max(eps, (np.sum(abs(ga_W[l])) + np.sum(abs(gn_W[len(self.W)-1-l]))))
            diff_b = np.sum(abs(ga_b[l] - gn_b[len(self.b)-1-l])) / max(eps, (np.sum(abs(ga_b[l])) + np.sum(abs(gn_b[len(self.b)-1-l]))))

            print("------- Relative error  -------")
            print('diff W  ' + str(diff_W))
            print('diff b  ' + str(diff_b))

    def ComputeGradsNum(self, X, Y, h):
        """
        [_,c] = self.ComputeCost(X, Y);
        grads_W = []
        grads_b = []

        for l in range(len(self.b)):

            grad_b = (np.zeros((len(self.b[l]), 1)))

            for i in range(len(self.b[l])):
                b_try_layer = np.array(self.b[l])
                b_try_layer[i] += h
                self.b[l] = b_try_layer
                [_,c2] = self.ComputeCost(X, Y)
                grad_b[i] = (c2-c) / h

            grads_b.append(grad_b)

            print("Gradients of b at layer " + str(l) + "  is done....")


        print("Gradients of b is done....")

        for l in range(len(self.W)):

            grad_W = np.zeros(self.W[l].shape)

            for i in range(self.W[l].shape[0]):
                for j in range(self.W[l].shape[1]):
                    W_try_layer = np.array(self.W[l])
                    W_try_layer[i,j] += h
                    self.W[l] = W_try_layer
                    [_,c2] = self.ComputeCost(X, Y)
                    #print(str(c2) + "   " + str(c) + " = " + str((c2-c)/ h))
                    grad_W[i,j] = (c2-c) / h

            grads_W.append(grad_W)

            print("Gradients of W at layer " + str(l) + "  is done....")

        return [grads_W, grads_b]

        """

        grad_W = []
        grad_B = []
        for l in range(self.k):
            grad_W.append(np.zeros(self.W[l].shape))

            for i in range(len(self.W[l].flatten())):
                old_par = self.W[l].flat[i]
                self.W[l].flat[i] = old_par + h
                _, c2 = self.ComputeCost(X, Y)
                self.W[l].flat[i] = old_par - h
                _, c3 = self.ComputeCost(X, Y)
                self.W[l].flat[i] = old_par
                grad_W[l].flat[i] = (c2 - c3) / (2 * h)

        for l in range(self.k):
            grad_B.append(np.zeros(self.b[l].shape))

            for i in range(len(self.b[l].flatten())):
                old_par = self.b[l].flat[i]
                self.b[l].flat[i] = old_par + h
                _, c2 = self.ComputeCost(X, Y)
                self.b[l].flat[i] = old_par - h
                _, c3 = self.ComputeCost(X, Y)
                self.b[l].flat[i] = old_par
                grad_B[l].flat[i] = (c2 - c3) / (2 * h)

        return grad_W, grad_B

    def ComputeGradients(self, X, Y):

        grad_W = []
        grad_b = []
        n = X.shape[1]

        [P, x] = self.EvaluateClassifier(X)
        G = -(Y - P)

        for l in range(self.k-1,0,-1):
            grad_W.append(((1/n) * np.dot(G, x[l-1].transpose())) + 2 * self.lamb * self.W[l])
            grad_b.append((1/n) * np.dot(G, np.ones((n,1))))

            G = np.dot(self.W[l].transpose(), G)
            G = np.multiply(G, np.sign(np.maximum(0, x[l-1])))

        grad_W.append(((1/n) * np.dot(G, X.transpose())) + 2 * self.lamb * self.W[0])
        grad_b.append((1/n) * np.dot(G, np.ones((n,1))))

        return grad_W, grad_b

    def EvaluateClassifier(self, X):

        x = []
        x.append(np.maximum(0, np.dot(self.W[0],X) + self.b[0])) # Relu ectivation function

        for l in range(1, self.k):
            x.append(np.maximum(0, np.dot(self.W[l], x[l-1]) + self.b[l])) # Relu ectivation function

        P = softmax(np.dot(self.W[self.k-1], x[self.k-2]) + self.b[self.k-1])

        return P, x


    def ComputeCost(self, X, Y):
        n = X.shape[1]
        [P, x] = self.EvaluateClassifier(X)
        loss = - Y * np.log(P)
        loss = (np.sum(loss) / n)
        regularization = self.lamb * np.sum(power(self.W))
        J = loss + regularization

        return J, loss

    def ComputeAccuracy(self, X, Y):
        [P, x] = self.EvaluateClassifier(X)
        P = np.argmax(P,0)
        Y = np.argmax(Y,0)
        acc = np.sum(P == Y) / len(Y)

        return acc

    def MiniBatchGD(self):
        costs_train = []
        costs_valid = []
        losses_train = []
        losses_valid = []

        n = self.trainX.shape[1]
        l = 0 # cycles
        t = 0
        k = 0
        etas = []
        accs_train = []
        accs_valid = []
        updateSteps = self.GDparams['n_epochs'] * self.GDparams['n_batch']

        for i in range(self.GDparams['n_epochs']):
            for j in range(1, int(n / self.GDparams['n_batch'])):
                j_start = (j-1) * self.GDparams['n_batch']
                j_end   = j * self.GDparams['n_batch']
                batch_X = self.trainX[:, j_start:j_end]
                batch_Y = self.trainY[:, j_start:j_end]

                [gn_W, gn_b] = self.ComputeGradients(batch_X, batch_Y)

                t += 1
                l = (t // (self.GDparams['n_s'] * 2))


                if(2*l*self.GDparams['n_s'] <= t and t <= (2*l+1)*self.GDparams['n_s']):
                    eta = self.GDparams['eta_min']+(((t-2*l*self.GDparams['n_s'])/self.GDparams['n_s'])*(self.GDparams['eta_max']-self.GDparams['eta_min']))

                elif((2*l+1)*self.GDparams['n_s'] <= t and t <= 2*(l+1)*self.GDparams['n_s']):
                    eta = self.GDparams['eta_max']-(((t-(2*l+1)*self.GDparams['n_s'])/self.GDparams['n_s'])*(self.GDparams['eta_max']-self.GDparams['eta_min']))

                etas.append(eta)

                for layer in range(len(self.W)):
                    self.W[layer] -= eta*gn_W[len(self.W)-1-layer]
                    self.b[layer] -= eta*gn_b[len(self.b)-1-layer]

                k += 1
                if(k == 500):

                    [cost_train, loss_train] = self.ComputeCost(self.trainX, self.trainY)
                    [cost_valid, loss_valid] = self.ComputeCost(self.validationX, self.validationY)
                    acc_train  = self.ComputeAccuracy(self.trainX, self.trainY)
                    acc_valid  = self.ComputeAccuracy(self.validationX, self.validationY)
                    costs_train.append(cost_train)
                    costs_valid.append(cost_valid)
                    losses_train.append(loss_train)
                    losses_valid.append(loss_valid)
                    accs_train.append(acc_train)
                    accs_valid.append(acc_valid)
                    etas.append(eta)

                    k = 0
                    print('Train Accuracy: ' + str(acc_train) + '    Valid Accuracy: ' + str(acc_valid))

            print(str(i+1) + ' of ' + str(self.GDparams['n_epochs']))

        acc_test = self.ComputeAccuracy(self.testX, self.testY)
        print('Test Accuracy: ' + str(acc_test) +'  Lambda: ' + str(self.lamb))

        self.plotAcc(accs_train, accs_valid, acc_test)
        self.plotCosts(costs_train, costs_valid)
        self.plotLoss(losses_train, losses_valid)
        self.plotEtas(etas)


    def plotAcc(self, accs_train, accs_valid, acc_test):
        t = np.arange(0, len(accs_train)*500, 500)
        fig, ax = plt.subplots()
        ax.plot(t, accs_train, label = 'Training')
        ax.plot(t, accs_valid, label = 'Validation')
        ax.legend()
        ax.set(xlabel='Update steps', ylabel='Accuracy', title='Lambda = ' + str(self.lamb) + ' Test Accuracy: ' + str(acc_test))
        plt.show()

    def plotCosts(self, costs_train, costs_valid):
        t = np.arange(0, len(costs_train)*500, 500)
        fig, ax = plt.subplots()
        ax.plot(t, costs_train, label = 'Training')
        ax.plot(t, costs_valid, label = 'Validation')
        ax.legend()
        ax.set(xlabel='Update steps', ylabel='Cost',  title='Lambda = ' + str(self.lamb))
        plt.show()

    def plotLoss(self, loss_train, loss_valid):
        t = np.arange(0, len(loss_train)*500, 500)
        fig, ax = plt.subplots()
        train = ax.plot(t, loss_train, label = 'Training')
        valid = ax.plot(t, loss_valid, label = 'Validation')
        ax.legend()
        ax.plot(t, loss_train, t, loss_valid)
        ax.set(xlabel='Update steps', ylabel='Loss',  title='Lambda = ' + str(self.lamb))
        plt.show()

    def plotEtas(self, etas):
        t = np.arange(0, len(etas))
        fig, ax = plt.subplots()
        ax.plot(t, etas)
        ax.set(xlabel='Update steps', ylabel='eta', title='Lambda: ' + str(self.lamb))
        plt.show()

def power(myList):
    return [ np.sum(x**2) for x in myList ]

def getDataOriginal():
    np.random.seed(seed=400)
    training_batch = LoadBatch("data_batch_1")
    trainX = training_batch.get(b'data')
    trainy = training_batch.get(b'labels')

    validation_batch = LoadBatch("data_batch_2")
    validationX = validation_batch.get(b'data')
    validationy = validation_batch.get(b'labels')

    # VALIDATION
    test_batch = LoadBatch("test_batch")
    testX = test_batch.get(b'data')
    testy = test_batch.get(b'labels')

    # Transpose
    trainX = trainX.transpose()
    validationX = validationX.transpose()
    testX = testX.transpose()

    # Create one hot matrix
    trainY = np.zeros((len(trainy), max(trainy)+1))
    trainY[np.arange(len(trainy)),trainy] = 1
    trainY = trainY.transpose()

    validationY = np.zeros((len(validationy), max(validationy)+1))
    validationY[np.arange(len(validationy)),validationy] = 1
    validationY = validationY.transpose()

    testY = np.zeros((len(testy), max(testy)+1))
    testY[np.arange(len(testy)),testy] = 1
    testY = testY.transpose()

    # Normalizing
    mean_X = np.mean(trainX, 1)
    std_X  = np.std(trainX, 1)

    trainX = trainX.transpose() - mean_X
    trainX = trainX / std_X
    testX = testX.transpose() - mean_X
    testX = testX / std_X
    validationX = validationX.transpose() - mean_X
    validationX = validationX / std_X

    # Transpose once again
    trainX = trainX.transpose()
    validationX = validationX.transpose()
    testX = testX.transpose()

    #print(trainX.shape)
    #print(validationX.shape)
    #print(testX.shape)

    return trainX, trainY, validationX, validationY, testX, testY

def getData():
    np.random.seed(seed=400)

    #TRAINING

    training_batch = LoadBatch("data_batch_1")
    trainX = training_batch.get(b'data').transpose()
    trainy = training_batch.get(b'labels')

    training_batch_2 = LoadBatch("data_batch_2")
    trainX_2 = training_batch_2.get(b'data').transpose()
    trainy_2 = training_batch_2.get(b'labels')

    training_batch_3 = LoadBatch("data_batch_3")
    trainX_3 = training_batch_3.get(b'data').transpose()
    trainy_3 = training_batch_3.get(b'labels')

    training_batch_4 = LoadBatch("data_batch_4")
    trainX_4 = training_batch_4.get(b'data').transpose()
    trainy_4 = training_batch_4.get(b'labels')

    training_batch_5 = LoadBatch("data_batch_5")
    trainX_5 = training_batch_5.get(b'data').transpose()
    trainy_5 = training_batch_5.get(b'labels')


    trainX = np.concatenate((trainX[:,0:9000],trainX_2[:,0:9000],trainX_3[:,0:9000],trainX_4[:,0:9000],trainX_5[:,0:9000]), axis=1)


    # VALIDATION

    validation_batch = LoadBatch("data_batch_1")
    validationX = validation_batch.get(b'data').transpose()
    validationy = validation_batch.get(b'labels')

    validation_batch_2 = LoadBatch("data_batch_2")
    validationX_2 = validation_batch_2.get(b'data').transpose()
    validationy_2 = validation_batch_2.get(b'labels')

    validation_batch_3 = LoadBatch("data_batch_3")
    validationX_3 = validation_batch_3.get(b'data').transpose()
    validationy_3 = validation_batch_3.get(b'labels')

    validation_batch_4 = LoadBatch("data_batch_4")
    validationX_4 = validation_batch_4.get(b'data').transpose()
    validationy_4 = validation_batch_4.get(b'labels')

    validation_batch_5 = LoadBatch("data_batch_5")
    validationX_5 = validation_batch_5.get(b'data').transpose()
    validationy_5 = validation_batch_5.get(b'labels')

    validationX = np.concatenate((validationX[:,9000:10000],validationX_2[:,9000:10000],validationX_3[:,9000:10000],validationX_4[:,9000:10000],validationX_5[:,9000:10000]), axis=1)

    # TEST

    test_batch = LoadBatch("test_batch")
    testX = test_batch.get(b'data').transpose()
    testy = test_batch.get(b'labels')

    # ONE HOT MATRIX
    trainY = np.zeros((len(trainy), max(trainy)+1))
    trainY[np.arange(len(trainy)),trainy] = 1
    trainY = trainY.transpose()

    trainY_2 = np.zeros((len(trainy_2), max(trainy_2)+1))
    trainY_2[np.arange(len(trainy_2)),trainy_2] = 1
    trainY_2 = trainY_2.transpose()

    trainY_3 = np.zeros((len(trainy_3), max(trainy_3)+1))
    trainY_3[np.arange(len(trainy_3)),trainy_3] = 1
    trainY_3 = trainY_3.transpose()

    trainY_4 = np.zeros((len(trainy_4), max(trainy_4)+1))
    trainY_4[np.arange(len(trainy_4)),trainy_4] = 1
    trainY_4 = trainY_4.transpose()

    trainY_5 = np.zeros((len(trainy_5), max(trainy_5)+1))
    trainY_5[np.arange(len(trainy_5)),trainy_5] = 1
    trainY_5 = trainY_5.transpose()


    validationY = np.zeros((len(validationy), max(validationy)+1))
    validationY[np.arange(len(validationy)),validationy] = 1
    validationY = validationY.transpose()

    validationY_2 = np.zeros((len(validationy_2), max(validationy_2)+1))
    validationY_2[np.arange(len(validationy_2)),validationy_2] = 1
    validationY_2 = validationY_2.transpose()

    validationY_3 = np.zeros((len(validationy_3), max(validationy_3)+1))
    validationY_3[np.arange(len(validationy_3)),validationy_3] = 1
    validationY_3 = validationY_3.transpose()

    validationY_4 = np.zeros((len(validationy_4), max(validationy_4)+1))
    validationY_4[np.arange(len(validationy_4)),validationy_4] = 1
    validationY_4 = validationY_4.transpose()

    validationY_5 = np.zeros((len(validationy_5), max(validationy_5)+1))
    validationY_5[np.arange(len(validationy_5)),validationy_5] = 1
    validationY_5 = validationY_5.transpose()


    testY = np.zeros((len(testy), max(testy)+1))
    testY[np.arange(len(testy)),testy] = 1
    testY = testY.transpose()


    trainY = np.concatenate((trainY[:,0:9000],trainY_2[:,0:9000],trainY_3[:,0:9000],trainY_4[:,0:9000],trainY_5[:,0:9000]), axis=1)
    validationY = np.concatenate((validationY[:,9000:10000],validationY_2[:,9000:10000],validationY_3[:,9000:10000],validationY_4[:,9000:10000],validationY_5[:,9000:10000]), axis=1)

    # Normalizing
    mean_X = np.mean(trainX, 1)
    std_X  = np.std(trainX, 1)

    trainX = trainX.transpose() - mean_X
    trainX = trainX / std_X
    testX = testX.transpose() - mean_X
    testX = testX / std_X
    validationX = validationX.transpose() - mean_X
    validationX = validationX / std_X

    # Transpose once again
    trainX = trainX.transpose()
    validationX = validationX.transpose()
    testX = testX.transpose()

    return trainX, trainY, validationX, validationY, testX, testY

# -----------------

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('Dataset/'+ filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dict[b'data'] = dict[b'data'].astype(float)
    return dict

def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


def main():

    rnn = RNN()
    #rnn.check_gradients()

    rnn.MiniBatchGD()

if __name__ == "__main__":
    main()
