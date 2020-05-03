import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

def ComputeGradsNum(X, Y, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
    	for j in range(W.shape[1]):
    		W_try = np.array(W)
    		W_try[i,j] += h
    		c2 = ComputeCost(X, Y, W_try, b, lamda)
    		grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W1, W2, b1, b2, lamda, h):
    """ Converted from matlab code """
    grad_W1 = np.zeros((W1.shape[0], W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0], W2.shape[1]))
    grad_b1 = np.zeros((b1.shape[0], b1.shape[1]))
    grad_b2 = np.zeros((b2.shape[0], b2.shape[1]))

    for i in range(len(b1)):
    	b1_try = np.array(b1)
    	b1_try[i] -= h
    	c1 = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)

    	b1_try = np.array(b1)
    	b1_try[i] += h
    	c2 = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)

    	grad_b1[i] = (c2-c1) / (2*h)

    for i in range(len(b2)):
    	b2_try = np.array(b2)
    	b2_try[i] -= h
    	c1 = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)

    	b2_try = np.array(b2)
    	b2_try[i] += h
    	c2 = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)

    	grad_b2[i] = (c2-c1) / (2*h)


    for i in range(W1.shape[0]):
    	for j in range(W1.shape[1]):
    		W1_try = np.array(W1)
    		W1_try[i,j] -= h
    		c1 = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)

    		W1_try = np.array(W1)
    		W1_try[i,j] += h
    		c2 = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)

    		grad_W1[i,j] = (c2-c1) / (2*h)

    for i in range(W2.shape[0]):
    	for j in range(W2.shape[1]):
    		W2_try = np.array(W2)
    		W2_try[i,j] -= h
    		c1 = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)

    		W2_try = np.array(W2)
    		W2_try[i,j] += h
    		c2 = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)

    		grad_W2[i,j] = (c2-c1) / (2*h)

    return [grad_W1, grad_b1, grad_W2, grad_b2]

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

def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio
    sio.savemat(name + '.mat', {name:b})

# ----------- My Own down below ---------------

def EvaluateClassifier(X, W1, W2, b1, b2):
    s1 = np.dot(W1,X) + b1
    h = np.maximum(0,s1) # Relu ectivation function
    s = np.dot(W2,h) + b2
    P = softmax(s)
    return P, h

def ComputeCost(X, Y, W1, W2, b1, b2, lamb):
    [P, h] = EvaluateClassifier(X, W1, W2, b1, b2)
    n = X.shape[1]
    loss = - Y * np.log(P)
    loss = (np.sum(loss) / n)
    regularization = lamb * (np.sum(W1**2) + np.sum(W2**2))
    J = loss + regularization

    return J, loss

def ComputeAccuracy(X, Y, W1, W2, b1, b2): # Should be 10 % if not trained since there are 10 classes
    [P, h] = EvaluateClassifier(X, W1, W2, b1, b2)
    P = np.argmax(P,0)
    Y = np.argmax(Y,0)
    acc = np.sum(P == Y) / len(Y)

    return acc

def ComputeGradients(X, Y, W1, W2, b1, b2, lamb):
    m = W1.shape[0]
    K = W2.shape[0]
    d = X.shape[1]
    n = X.shape[1]

    [P, h] = EvaluateClassifier(X, W1, W2, b1, b2)
    G = -(Y - P)

    grad_W2 = ((1/n) * np.dot(G, h.transpose())) + 2 * lamb * W2
    grad_b2 = (1/n) * np.dot(G, np.ones((n,1)))

    G = np.dot(W2.transpose(), G)
    G = np.multiply(G,np.sign(np.maximum(0,h)))

    grad_W1 = ((1/n) * np.dot(G, X.transpose())) + 2 * lamb * W1
    grad_b1 = (1/n) * np.dot(G, np.ones((n,1)))

    return grad_W1, grad_b1, grad_W2, grad_b2

def MiniBatchGD(trainX, trainY, validationX, validationY, testX, testY, GDparams, W1, W2, b1, b2, lamb):
    costs_train = []
    costs_valid = []
    losses_train = []
    losses_valid = []

    n = trainX.shape[1]
    l = 0 # cycles
    t = 0
    etas = []
    accs_train = []
    accs_valid = []

    for i in range(GDparams['n_epochs']):
        for j in range(1, int(n / GDparams['n_batch'])): # Number of batches
            j_start = (j-1) * GDparams['n_batch']
            j_end   = j * GDparams['n_batch']
            batch_X = trainX[:, j_start:j_end]
            batch_Y = trainY[:, j_start:j_end]

            t += 1
            if(t%(2*GDparams['n_s']) == 0):
                l += 1

            if(2*l*GDparams['n_s'] <= t and t <= (2*l+1)*GDparams['n_s']):
                eta = GDparams['eta_min']+(((t-2*l*GDparams['n_s'])/GDparams['n_s'])*(GDparams['eta_max']-GDparams['eta_min']))
            elif((2*l+1)*GDparams['n_s'] <= t and t <= 2*(l+1)*GDparams['n_s']):
                eta = GDparams['eta_max']-(((t-(2*l+1)*GDparams['n_s'])/GDparams['n_s'])*(GDparams['eta_max']-GDparams['eta_min']))

            etas.append(eta)

            [ga_W1, ga_b1, ga_W2, ga_b2] = ComputeGradients(batch_X, batch_Y, W1, W2, b1, b2, lamb)
            W1 -= eta*ga_W1
            W2 -= eta*ga_W2
            b1 -= eta*ga_b1
            b2 -= eta*ga_b2

            [cost_train, loss_train] = ComputeCost(trainX, trainY, W1, W2, b1, b2, lamb)
            [cost_valid, loss_valid] = ComputeCost(validationX, validationY, W1, W2, b1, b2, lamb)
            costs_train.append(cost_train)
            costs_valid.append(cost_valid)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            acc_train  = ComputeAccuracy(trainX, trainY, W1, W2, b1, b2)
            acc_valid  = ComputeAccuracy(validationX, validationY, W1, W2, b1, b2)

            accs_train.append(acc_train)
            accs_valid.append(acc_valid)


        print(str(i+1) + ' of ' + str(GDparams['n_epochs']))
        print('Train Accuracy: ' + str(acc_train) + '    Valid Accuracy: ' + str(acc_valid))

        #cost_train = ComputeCost(trainX, trainY, W1, W2, b1, b2, lamb)
        #cost_valid = ComputeCost(validationX, validationY, W1, W2, b1, b2, lamb)
        #costs_train.append(cost_train)
        #costs_valid.append(cost_valid)

    # Run final test once we found lambda
    acc_test = ComputeAccuracy(testX, testY, W1, W2, b1, b2)
    acc_valid  = ComputeAccuracy(validationX, validationY, W1, W2, b1, b2)

    print('Validation Accuracy: ' + str(acc_valid) +'  Lambda: ' + str(lamb))
    print('Test Accuracy: ' + str(acc_test) +'  Lambda: ' + str(lamb))

    #plotCostsEpocs(GDparams, costs_train, costs_valid)

    plotAcc(GDparams, accs_train, accs_valid, lamb, acc_test)
    plotCosts(GDparams, costs_train, costs_valid, lamb)
    plotLoss(GDparams, losses_train, losses_valid, lamb)
    plotEtas(etas, lamb)

    return W1, W2, b1, b2

def plotCostsEpocs(GDparams, costs_train, costs_valid, lamb): # Epocs
    t = np.arange(0, len(costs_train), 1)
    fig, ax = plt.subplots()
    ax.plot(t, cost_train, label = 'Training')
    ax.plot(t, cost_valid, label = 'Validation')
    ax.legend()
    ax.plot(t, costs_train, t, costs_valid)
    ax.set(xlabel='epochs', ylabel='loss',  title='Lambda = ' + str(lamb))
    plt.show()

def plotAcc(GDparams, accs_train, accs_valid, lamb, acc_test): # Update steps
    t = np.arange(0, len(accs_train), 1)
    fig, ax = plt.subplots()
    ax.plot(t, accs_train, label = 'Training')
    ax.plot(t, accs_valid, label = 'Validation')
    ax.legend()
    ax.set(xlabel='Update steps', ylabel='Accuracy', title='Lambda = ' + str(lamb) + ' Test Accuracy: ' + str(acc_test))
    plt.show()

def plotCosts(GDparams, costs_train, costs_valid, lamb): # Update steps
    t = np.arange(0, len(costs_train), 1)
    fig, ax = plt.subplots()
    ax.plot(t, costs_train, label = 'Training')
    ax.plot(t, costs_valid, label = 'Validation')
    ax.legend()
    ax.set(xlabel='Update steps', ylabel='Cost',  title='Lambda = ' + str(lamb))
    plt.show()

def plotLoss(GDparams, loss_train, loss_valid, lamb): # Update steps
    t = np.arange(0, len(loss_train), 1)
    fig, ax = plt.subplots()
    train = ax.plot(t, loss_train, label = 'Training')
    valid = ax.plot(t, loss_valid, label = 'Validation')
    ax.legend()
    ax.plot(t, loss_train, t, loss_valid)
    ax.set(xlabel='Update steps', ylabel='Loss',  title='Lambda = ' + str(lamb))
    plt.show()

def plotEtas(etas, lamb):
    t = np.arange(0, len(etas), 1)
    fig, ax = plt.subplots()
    ax.plot(t, etas)
    ax.set(xlabel='Update steps', ylabel='eta', title='Lambda: ' + str(lamb))
    plt.show()

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
    trainX_2 = training_batch.get(b'data').transpose()
    trainy_2 = training_batch.get(b'labels')

    training_batch_3 = LoadBatch("data_batch_3")
    trainX_3 = training_batch.get(b'data').transpose()
    trainy_3 = training_batch.get(b'labels')

    training_batch_4 = LoadBatch("data_batch_4")
    trainX_4 = training_batch.get(b'data').transpose()
    trainy_4 = training_batch.get(b'labels')

    training_batch_5 = LoadBatch("data_batch_5")
    trainX_5 = training_batch.get(b'data').transpose()
    trainy_5 = training_batch.get(b'labels')


    trainX = np.concatenate((trainX[:,0:9000],trainX_2[:,0:9000],trainX_3[:,0:9000],trainX_4[:,0:9000],trainX_5[:,0:9000]), axis=1)


    # VALIDATION

    validation_batch = LoadBatch("data_batch_1")
    validationX = validation_batch.get(b'data').transpose()
    validationy = validation_batch.get(b'labels')

    validation_batch_2 = LoadBatch("data_batch_2")
    validationX_2 = validation_batch.get(b'data').transpose()
    validationy_2 = validation_batch.get(b'labels')

    validation_batch_3 = LoadBatch("data_batch_3")
    validationX_3 = validation_batch.get(b'data').transpose()
    validationy_3 = validation_batch.get(b'labels')

    validation_batch_4 = LoadBatch("data_batch_4")
    validationX_4 = validation_batch.get(b'data').transpose()
    validationy_4 = validation_batch.get(b'labels')

    validation_batch_5 = LoadBatch("data_batch_5")
    validationX_5 = validation_batch.get(b'data').transpose()
    validationy_5 = validation_batch.get(b'labels')

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

def main():

    # X = d x n
    # Y = K x n
    # W1 = m x d
    # W2 = K x m
    # b1 = m x 1
    # b2 = K x 1
    # P = K x n

    [trainX, trainY, validationX, validationY, testX, testY] = getData()
    #[trainX, trainY, validationX, validationY, testX, testY] = getDataOriginal()

    d = trainX.shape[0] # Number of dimensions
    n = trainX.shape[1] # Number of images
    K = trainY.shape[0] # Number of classes
    m = 50 # nodes in the hidden layer


      # Cycles =  steps / (2*n_s)
    # steps = 'Number of batches' * 'epocs'
    # Number of batch = n / 'n_batch'
    #The positive integer ns is known as the stepsize and is usually chosen so that one cycle of training corresponds to a multiple of epochs of training.

    # n_s = 2 * floor(n / n_batch)

    # cycles = ( (n / n_batch) * epochs) / (2 * n_s)

    # cycles =  (n * epochs) / (2 * n_batch * n_s)

    n_s = 2 * (n / 100)
    print('n_s:  ' + str(n_s))
    print('n: ' + str(n))

    GDparams = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1, 'n_s': n_s, 'n_epochs': 12}
    lamb = 0.006551518774129204
    W1 = np.random.normal(0, 1/np.sqrt(d), (m,d))
    W2 = np.random.normal(0, 1/np.sqrt(m), (K,m))
    b1 = np.zeros((m,1))
    b2 = np.zeros((K,1))

    [W1, W2, b1, b2] = MiniBatchGD(trainX, trainY, validationX, validationY, testX, testY, GDparams, W1, W2, b1, b2, lamb)

    # Broad search
    #l_min = -5
    #l_max = -1

    # Narrow search
    """
    l_min = -3
    l_max = -2

    l = l_min + (l_max - l_min) * np.random.uniform(0, 1, 8)

    for lamb in l:
        [W1, W2, b1, b2] = MiniBatchGD(trainX, trainY, validationX, validationY, testX, testY, GDparams, W1, W2, b1, b2, 10**lamb)
    """

    """
    [ga_W1, ga_b1, ga_W2, ga_b2] = ComputeGradsNumSlow(trainX[:,0:100], trainY[:,0:100], W1, W2, b1, b2, lamb, 1e-5)
    [gn_W1, gn_b1, gn_W2, gn_b2] = ComputeGradients(trainX[:,0:100], trainY[:,0:100], W1, W2, b1, b2, lamb)

    print('SUMS:')
    print(np.sum(gn_W1))
    print(np.sum(ga_W1))
    print(np.sum(gn_W2))
    print(np.sum(ga_W2))
    print(np.sum(gn_b1))
    print(np.sum(ga_b1))
    print(np.sum(gn_b2))
    print(np.sum(ga_b2))

    eps = 0.000000000001
    diff_W1 = np.sum(abs(ga_W1 - gn_W1)) / max(eps, (np.sum(abs(ga_W1)) + np.sum(abs(gn_W1))))
    diff_b1 = np.sum(abs(ga_b1 - gn_b1)) / max(eps, (np.sum(abs(ga_b1)) + np.sum(abs(gn_b1))))

    diff_W2 = np.sum(abs(ga_W2 - gn_W2)) / max(eps, (np.sum(abs(ga_W2)) + np.sum(abs(gn_W2))))
    diff_b2 = np.sum(abs(ga_b2 - gn_b2)) / max(eps, (np.sum(abs(ga_b2)) + np.sum(abs(gn_b2))))
    print("------- Relative error  -------")
    print('diff W1 ' + str(diff_W1))
    print('diff b1 ' + str(diff_b1))

    print('diff W2 ' + str(diff_W1))
    print('diff b2 ' + str(diff_b1))
    """

if __name__ == "__main__":
    main()
