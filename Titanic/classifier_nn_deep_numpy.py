import copy
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

def main():
    X, Y = preprocess_data()
    cache = initialize_parameters(X, Y)
    
    for i in range(cache['iterationsCount']): # Training iterations
        logger = initialize_system()
        cache['currentIterationNumber'] = i
        make_forward_propagation(cache)
        compute_cost(cache, logger)
        make_backward_propagation(cache)
        update_parameters(cache)

    estimate_training_set_accuracy(cache, logger)

def initialize_system():
    logName = 'Logs/Log_' + str(datetime.datetime.now().strftime('%Y_%m%d_%H%M_%S%f')) + '.txt'
    logger = logging.getLogger('logger')
    logging.basicConfig(filename=logName, level=logging.INFO)
    return logger

def preprocess_data():
    rawData = pd.read_csv('train.csv')
    preProcessedData = np.array(copy.deepcopy(rawData))

    # In Sex column, male/female converted to 1/0, respectively
    preProcessedData[:, 4] = (preProcessedData[:, 4] == 'male').astype(np.float32) # In Sex column, male/female converted to 1/0, respectively
    
    # In Cabin column, with/without data converted to 1/0, respectively
    preProcessedData[:, 10] = pd.notnull(preProcessedData[:, 10]).astype(np.float32) # In Cabin column, with/without data converted to 1/0, respectively

    # In Age column, Age normalization, and assign 0 to nan age entries
    ages = preProcessedData[:, 5].astype(np.float32)
    ageAverage = np.nanmean(ages)
    ageStdDev = np.nanstd(ages)
    preProcessedData[:, 5] = np.divide(ages - ageAverage, ageStdDev) + 1
    preProcessedData[np.where(pd.isnull(preProcessedData[:, 5])), 5] = 0

    # In Fare column, Fare normalzation
    fares = preProcessedData[:, 9].astype(np.float32)
    fareAverage = np.nanmean(fares)
    fareStdDev = np.nanstd(fares)
    preProcessedData[:, 9] = np.divide(fares - fareAverage, fareStdDev) + 1

    # In Embarked column, letters converted to digits
    preProcessedData[np.where(pd.isnull(preProcessedData[:, 11])), 11] = 2
    preProcessedData[np.where(preProcessedData[:, 11] == 'C'), 11] = 1
    preProcessedData[np.where(preProcessedData[:, 11] == 'Q'), 11] = -1
    preProcessedData[np.where(preProcessedData[:, 11] == 'S'), 11] = 0

    X = preProcessedData[:, [2, 4, 5, 6, 7, 9, 10, 11]].T # Pick selected 8 columns for training (has to drop column 0 because huge numbers (for example, 891) will easily overflow the sigmoid evaluation)
    Y = preProcessedData[:, [1]].T

    return X, Y
    
def initialize_parameters(X, Y):
    np.random.seed(datetime.datetime.now().microsecond)
    m = X.shape[1]
    N = [X.shape[0], 32, 8, Y.shape[0]]
    L = len(N) - 1 # neural network layers count. Minus one to exclude input layer
    A = [X] # is regarded as A0, which doesn't count for number of Layers

    # Assign dummy empty array as first item of the list so that index "1" will point to items of Layer One
    B = [np.array([])] # Biases
    dB = [np.array([])] # dJ/dB
    dW = [np.array([])] # dJ/dW
    SdB = [np.array([])]  # SdB for ADAM optimization
    SdW = [np.array([])]  # SdW for ADAM optimization
    VdB = [np.array([])]  # VdB for ADAM optimization
    VdW = [np.array([])]  # VdW for ADAM optimization
    W = [np.array([])] # Weights
    Z = [np.array([])]
    zAverage = [np.array([])]
    zStdDev = [np.array([])]

    for i in range(L):
        A.append(np.zeros((N[i+1], m)))
        B.append(np.zeros((N[i+1], 1)))
        W.append(np.random.randn(N[i+1], N[i]) * np.power(10., -2)) # Randomly initializing neuros. Factor 0.01 is to ensure the starting parameter to be small to stay on linear zone (crucial for gradiant decent on sigmoid)
        dB.append(np.zeros((B[i+1].shape)))
        dW.append(np.zeros((W[i+1].shape)))
        SdB.append(np.zeros((B[i+1].shape)))
        SdW.append(np.zeros((W[i+1].shape)))
        VdB.append(np.zeros((B[i+1].shape)))
        VdW.append(np.zeros((W[i+1].shape)))
        Z.append(np.zeros((N[i+1], m)))
        zAverage.append(np.zeros((N[i+1], 1)))
        zStdDev.append(np.zeros((N[i+1], 1)))

    cache = {
        'A': A,
        'B': B,
        'dB': dB,
        'dW': dW,
        'iterationsCount': np.power(10, 5) + 1,
        'L': L,
        'm': m,
        'N': N,
        'SdB': SdB,
        'SdW': SdW,
        'VdB': VdB,
        'VdW': VdW,
        'W': W,
        'X': X,
        'Y': Y,
        'Z': Z,
        'zAverage': zAverage,
        'zStdDev': zStdDev,
        'alpha': 6 * np.power(10., -3), # Learning rate
        'alphaDecay': 1 - np.power(10., -5), # decay rate on Learning Rate 
        'beta1': 0.93, # Hyperparameter for the moment factor used in ADAM optimization
        'beta2': 0.999, # Hyperparameter for the RMS Prop factor used in ADAM optimization
        'epsilon': np.power(10., -8) # Random small constant for axilliary use
    }
    
    return cache

# Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
def make_forward_propagation(cache):
    A = cache['A']
    B = cache['B']
    L = cache['L']
    W = cache['W']
    Z = cache['Z']
    zAverage = cache['zAverage']
    zStdDev = cache['zStdDev']
    
    for i in range(L):
        Z[i+1] = (np.dot(W[i+1], A[i]) + B[i+1]).astype(np.float32) # Z_l = W_l * A_l-1 + B_l
        
        if (i < L - 1): # For all hidden layers (Layer 1 ~ L-1)
            zAverage[i+1] = np.array(np.average(Z[i+1], axis=1)).reshape(Z[i+1].shape[0], 1)
            zStdDev[i+1] = np.array(np.std(Z[i+1], axis=1)).reshape(Z[i+1].shape[0], 1)

            Z[i+1] = np.divide(Z[i+1] - zAverage[i+1], zStdDev[i+1]) + 1 # Z Normalization to facilitate learning

            # Optional verification to ensure Z entries collectively have average 0 and stdDev 1
            # assert(np.average(np.average(Z[i+1], axis=1)) - 1 < np.sqrt(cache['epsilon']))
            # assert(np.average(np.std(Z[i+1], axis=1)) - 1 < np.sqrt(cache['epsilon']))

            A[i+1] = np.maximum(0, Z[i+1]) # Apply ReLU function max(0, Z)
        else: # For output layer (Layer L)
            A[L] = expit(Z[L]) # Apply sigmoid function 1 / (1 + e^-Z). Used to mitigate overflow of np.exp(). See http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html

    cache['A'] = A
    cache['Z'] = Z
    cache['zAverage'] = zAverage # To be used in prediction
    cache['zStdDev'] = zStdDev # To be used in prediction

# Implement the cost function defined per Cross Entropy Cost function. See https://en.wikipedia.org/wiki/Cross_entropy
def compute_cost(cache, logger): # Only compute top layer cost
    currentIterationNumber = cache['currentIterationNumber']
    
    if(np.remainder(currentIterationNumber, 100) == 0):
        A = cache['A']
        B = cache['B']
        iterationsCount = cache['iterationsCount']
        L = cache['L']
        m = cache['m']
        W = cache['W']
        Y = cache['Y']
        epsilon = cache['epsilon']
        
        firstTerm = np.dot(Y, np.log(A[L] + epsilon).T) # epsilon to avoid log(0) exception.
        secondTerm = np.dot(1-Y, np.log(1 - A[L] + epsilon).T) # epsilon to avoid log(0) exception.
        cost = np.squeeze(-np.divide(firstTerm + secondTerm, m)) # np.squeeze to turn one cell array into scalar
        print('#' + str(currentIterationNumber).rjust(8, '0') + ': ' + str(cost).ljust(20, ' ') + ' ' + str(cache['alpha']))
        logger.info('#' + str(currentIterationNumber).rjust(8, '0') + ': ' + str(cost).ljust(20, ' ') + ' ' + str(cache['alpha'])) # Log progress with Cost

        if(currentIterationNumber == iterationsCount - 1): # Final W's and B's are needed for prediction
            for i in range(L):
                logger.info(W[i + 1])
                logger.info(B[i + 1])

# Implement the backward propagation for the SIGMOID -> LINEAR -> [LINEAR->RELU] * (L-1) 
# Implemented references:
# 1. https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%2Byour%2BDeep%2BNeural%2BNetwork%2B-%2BStep%2Bby%2BStep%2Bv3.ipynb
# 2. https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils_v2.py
def make_backward_propagation(cache):
    grads = {} # Only carry dA's, since they are not needed in forward-props
    A = cache['A']
    L = cache['L']
    m = cache['m']
    Y = cache['Y']
    Z = cache['Z']
    W = cache['W']
    dW = cache['dW']
    dB = cache['dB']
    epsilon = cache['epsilon']

    # Sigmoid backward-prop implementation. Only for output layer
    dAL = -np.divide(Y, A[L]+epsilon) + np.divide(1-Y, 1-A[L]+epsilon)
    AL = expit(Z[L])
    dZ = dAL * AL * (1 - AL) # element-wise multiplication
    dW[L] = np.divide(np.dot(dZ, A[L-1].T), m)
    dB[L] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
    grads['dA' + str(L-1)] = np.dot(W[L].T, dZ)

    # ReLU backward-prop implementation. For all hidden layers
    for l in reversed(range(1, L)):
        dZ = np.array(grads['dA' + str(l)], copy=True)
        dZ[Z[l] <= 0] = 0 # essense of relu_backward(). When z <= 0, you should set dz to 0 as well. See https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils_v2.py
        dW[l] = np.divide(np.dot(dZ, A[l-1].T), m)
        dB[l] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
        grads['dA' + str(l-1)] = np.dot(W[l].T, dZ)

    cache['dW'] = dW
    cache['dB'] = dB

# Update parameters using ADAM
def update_parameters(cache):
    B = cache['B']
    L = cache['L']
    dW = cache['dW']
    dB = cache['dB']
    SdW = cache['SdW']
    SdB = cache['SdB']
    VdW = cache['VdW']
    VdB = cache['VdB']
    W = cache['W']
    alpha = cache['alpha']
    alphaDecay = cache['alphaDecay']
    beta1 = cache['beta1']
    beta2 = cache['beta2']
    epsilon = cache['epsilon']

    alpha *= alphaDecay # Gradually ramp down learning rate as closing in to min
    
    for i in range(L):
        VdW[i+1] = beta1 * VdW[i+1] + (1 - beta1) * dW[i+1]
        VdB[i+1] = beta1 * VdB[i+1] + (1 - beta1) * dB[i+1]
        SdW[i+1] = beta2 * SdW[i+1] + (1 - beta2) * np.power(dW[i+1], 2)
        SdB[i+1] = beta2 * SdB[i+1] + (1 - beta2) * np.power(dB[i+1], 2)
        wUpdateAddingRmsPropElement = np.sqrt((SdW[i+1] + epsilon).astype(np.float32))
        bUpdateAddingRmsPropElement = np.sqrt((SdB[i+1] + epsilon).astype(np.float32))
        wUpdateAddingMomentumElement = np.divide(VdW[i+1], wUpdateAddingRmsPropElement)
        bUpdateAddingMomentumElement = np.divide(VdB[i+1], bUpdateAddingRmsPropElement)
        wUpdateAddingAlpha = alpha * wUpdateAddingMomentumElement
        bUpdateAddingAlpha = alpha * bUpdateAddingMomentumElement
        
        W[i+1] = W[i+1] - wUpdateAddingAlpha
        B[i+1] = B[i+1] - bUpdateAddingAlpha

    cache['VdW'] = VdW
    cache['VdB'] = VdB
    cache['SdW'] = SdW
    cache['SdB'] = SdB
    cache['W'] = W
    cache['B'] = B
    cache['alpha'] = alpha

# Make rough estimation of accuracy on the training records with ages
def estimate_training_set_accuracy(cache, logger):
    A = cache['A']
    L = cache['L']
    Y = cache['Y']
    m = cache['m']

    logger.info('Rough accuracy on training set: ' + str(np.sum((np.abs(np.abs(A[L]) - Y) < 0.5).astype(int)) / float(m)))

main()