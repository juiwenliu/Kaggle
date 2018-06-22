import copy
import datetime
from io import StringIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.special import expit

def main():
    X, Y = preprocess_data()
    cache, logger = initialize_parameters(X, Y)

    for i in range(cache['iterationsCount']): # Training iterations
        make_forward_propagation(cache)
        compute_cost(cache)
        make_backward_propagation(cache)
        update_parameters(cache)

        if (np.remainder(i, 1000) == 0):
            logger.info(str(i).rjust(10) + ' ' + str(cache['cost'])) # Log progress with Cost
            print(str(i).rjust(10) + ' ' + str(cache['cost']))
            cache['alpha'] *= cache['alphaDecay'] # Gradually ramp down learning rate when closing in to min

            if (np.remainder(i, 100000) == 0):
                W = cache['W']
                B = cache['B']
                L = cache['L']

                for j in range(1, L+1):
                    logger.info(W[j])
                    logger.info(B[j])

    estimate_training_set_accuracy(cache)    
    
    # Final W's aWnd B's are needed for prediction
    logger.info('W: ' + str(cache['W']))
    logger.info('B: ' + str(cache['B']))

def preprocess_data():
    with open('train.csv','r') as f:
        rawRows = f.read()
        rawRowsLastNameCommaStripped = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRows) # drop comma between lastname and firstname

    preProcessedData = np.genfromtxt(StringIO(rawRowsLastNameCommaStripped), delimiter= ',', dtype='|U64') # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html Cast all cells to U64 string type to avoid structured ndarray (rank 1 array)
    preProcessedData = preProcessedData[1:] # Drop header row

    # Evaluate average age of known ages, and assign the same to those without known age
    counter = 0
    sum = 0

    for i in range(preProcessedData.shape[0]):
        if(preProcessedData[i, 5] != ''):
            sum += preProcessedData[i, 5].astype(np.float64)
            counter += 1

    ageAverage = np.divide(sum, counter)

    for i in range(preProcessedData.shape[0]):
        preProcessedData[i, 5] = ageAverage if (preProcessedData[i, 5] == '') else preProcessedData[i, 5]

    preProcessedData[:, 5] = np.divide(preProcessedData[:, 5].astype(np.float64), 10) # Age normalization to scale down age by one order of magnitude
    preProcessedData[:, 9] = np.divide(preProcessedData[:, 9].astype(np.float64), 10) # Fare normalization to scale down age by one order of magnitude
    preProcessedData[:, 4] = (preProcessedData[:, 4] == 'male').astype(int) # In Sex column, male/female converted to 1/0, respectively
    preProcessedData[:, 10] = (preProcessedData[:, 10] != '').astype(int) # In Cabin column, with/without data converted to 1/0, respectively
    preProcessedData = preProcessedData[:, [1, 2, 4, 5, 6, 7, 9, 10]] # Pick selected 8 columns for training (has to drop column 0 because huge numbers (for example, 891) will easily overflow the sigmoid evaluation)
    preProcessedData = preProcessedData.astype(np.float64) # Convert all cells to float.

    X = copy.deepcopy(preProcessedData[:, [1, 2, 3, 4, 5, 6, 7]]).T # Make deep copy to avoid corrupting raw data
    Y = copy.deepcopy(preProcessedData[:, 0]).reshape(X.shape[1], -1).T # Make deep copy to avoid corrupting raw data. Reshape to address rank-1 array issue
    return X, Y

def initialize_parameters(X, Y):
    m = X.shape[1] # Training Set count
    N = [
        X.shape[0], # Feature count
        64, # Layer 1 neural unit count
        64, # Layer 1 neural unit count
        8, # Layer 1 neural unit count
        8, # Layer 1 neural unit count
        1  # Layer 4 neural unit (output) count
    ]
    L = len(N) - 1 # neural network layers count. Minus one to exclude input layer
    W = [np.array([])] # empty array serves as dummy item
    B = [np.array([])] # empty array serves as dummy item
    dW = [np.array([])] # empty array serves as dummy item
    dB = [np.array([])] # empty array serves as dummy item
    VdW = [np.array([])] # empty array serves as dummy item. VdW for ADAM optimization
    VdB = [np.array([])] # empty array serves as dummy item. VdB for ADAM optimization
    SdW = [np.array([])] # empty array serves as dummy item. SdW for ADAM optimization
    SdB = [np.array([])] # empty array serves as dummy item. SdB for ADAM optimization
    Z = [np.array([])] # empty array serves as dummy item
    A = [X] # X is regarded as A0
    alpha = 3.446 * np.power(10., -4) # Learning rate
    beta1 = 0.962 # Exponential alphaDecay hyperparameter for the first moment estimates used in ADAM optimization
    beta2 = 0.99962 # Exponential alphaDecay hyperparameter for the second moment estimates used in ADAM optimization
    alphaDecay = 0.995 # decay rate on Learning Rate 
    epsilon = np.power(10., -7) # For divide-by-zero prevention and ADAM optimization
    iterationsCount = 5000000

    for i in range(1, L+1):
        np.random.seed(datetime.datetime.now().microsecond)
        W.append(np.random.randn(N[i], N[i-1]) * np.power(10., -2)) # Randomly initializing neuros. Factor 0.01 is to ensure the starting parameter to be small to stay on linear zone (crucial for gradiant decent on sigmoid)
        B.append(np.zeros((N[i], 1)))
        dW.append(np.zeros(W[i].shape))
        dB.append(np.zeros(B[i].shape))
        VdW.append(np.zeros(W[i].shape))
        VdB.append(np.zeros(B[i].shape))
        SdW.append(np.zeros(W[i].shape))
        SdB.append(np.zeros(B[i].shape))
        Z.append(np.zeros((N[i], m)))
        A.append(np.zeros(Z[i].shape))

    cache = {
        'X': X,
        'Y': Y,
        'm': m,
        'N': N,
        'L': L,
        'W': W,
        'B': B,
        'dW': dW,
        'dB': dB,
        'VdW': VdW,
        'VdB': VdB,
        'SdW': SdW,
        'SdB': SdB,
        'Z': Z,
        'A': A,
        'alpha': alpha,
        'beta1': beta1,
        'beta2': beta2,
        'alphaDecay': alphaDecay,
        'epsilon': epsilon,
        'iterationsCount': iterationsCount
    }

    logName = 'Logs/Log_' + str(datetime.datetime.now().strftime('%Y_%m%d_%H%M_%S%f')) + '.txt'
    logger = logging.getLogger('logger')
    logging.basicConfig(filename=logName, level=logging.INFO)
    logger.info('X: ' + str(X.shape))
    logger.info('Y: ' + str(Y.shape))
    logger.info('m: ' + str(m))
    logger.info('N: ' + str(N))
    logger.info('L: ' + str(L))

    for i in range(L+1):
        logger.info('W' + str(i) + ': ' + str(W[i].shape))
        logger.info('B' + str(i) + ': ' + str(B[i].shape))
        logger.info('dW' + str(i) + ': ' + str(dW[i].shape))
        logger.info('dB' + str(i) + ': ' + str(dB[i].shape))
        logger.info('VdW' + str(i) + ': ' + str(VdW[i].shape))
        logger.info('VdB' + str(i) + ': ' + str(VdB[i].shape))
        logger.info('SdW' + str(i) + ': ' + str(SdW[i].shape))
        logger.info('SdB' + str(i) + ': ' + str(SdB[i].shape))
        logger.info('Z' + str(i) + ': ' + str(Z[i].shape))
        logger.info('A' + str(i) + ': ' + str(A[i].shape))
    
    logger.info('alpha: ' + str(alpha))
    logger.info('beta1: ' + str(beta1))
    logger.info('beta2: ' + str(beta2))
    logger.info('alphaDecay: ' + str(alphaDecay))
    logger.info('epsilon: ' + str(epsilon))
    return cache, logger

# Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
def make_forward_propagation(cache):
    L = cache['L']
    W = cache['W']
    A = cache['A']
    B = cache['B']
    Z = cache['Z']

    for i in range(1, L): # For all hidden layers (Layer 1 ~ L-1)
        Z[i] = np.dot(W[i], A[i-1]) + B[i] # Z_l = W_l * A_l-1 + B_l
        A[i] = np.maximum(0, Z[i]) # Apply ReLU function max(0, Z)

    # For output layer (Layer L)
    Z[L] = np.dot(W[L], A[L-1]) + B[L] # Z_L = W_L * A_L-1 + B_L
    A[L] = expit(Z[L]) # Apply sigmoid function 1 / (1 + e^-Z). Used to mitigate overflow of np.exp(). See http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html
    cache['Z'] = Z
    cache['A'] = A

# Implement the cost function defined per Cross Entropy Cost function. See https://en.wikipedia.org/wiki/Cross_entropy
def compute_cost(cache): # Only compute top layer cost
    m = cache['m']
    A = cache['A']
    Y = cache['Y']
    L = cache['L']
    epsilon = cache['epsilon']
    firstTerm  = np.dot(  Y, np.log(  A[L]+epsilon).T) # epsilon to avoid log(0) exception.
    secondTerm = np.dot(1-Y, np.log(1-A[L]+epsilon).T) # epsilon to avoid log(0) exception
    cost = -(firstTerm + secondTerm) / m
    cache['cost'] = np.squeeze(cost) # np.squeeze to turn one cell array into scalar

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
    AL = expit(Z[L]) # Apply sigmoid function 1 / (1 + e^-Z). Used to mitigate overflow of np.exp(). See http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html
    dZ = dAL * AL * (1 - AL) # element-wise multiplication
    dW[L] = np.dot(dZ, A[L-1].T) / m
    dB[L] = np.sum(dZ, axis=1, keepdims=True) / m
    grads['dA' + str(L-1)] = np.dot(W[L].T, dZ)

    # Sigmoid backward-prop implementation. For all hidden layers
    for l in reversed(range(L-1)):
        dZ = np.array(grads['dA' + str(l+1)], copy=True) # just converting dz to a correct object.
        dZ[Z[l+1] <= 0] = 0 # essense of relu_backward(). When z <= 0, you should set dz to 0 as well. See https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils_v2.py
        dW[l+1] = np.dot(dZ, A[l].T) / m
        dB[l+1] = np.sum(dZ, axis=1, keepdims=True) / m
        grads['dA' + str(l)] = np.dot(W[l+1].T, dZ)

    cache['dW'] = dW
    cache['dB'] = dB

# Update parameters using ADAM
def update_parameters(cache):
    L = cache['L']
    W = cache['W']
    B = cache['B']
    dW = cache['dW']
    dB = cache['dB']
    VdW = cache['VdW']
    VdB = cache['VdB']
    SdW = cache['SdW']
    SdB = cache['SdB']
    alpha = cache['alpha']
    beta1 = cache['beta1']
    beta2 = cache['beta2']
    epsilon = cache['epsilon']

    for i in range(1, L+1):
        VdW[i] = beta1 * VdW[i] + (1 - beta1) * dW[i]
        VdB[i] = beta1 * VdB[i] + (1 - beta1) * dB[i]
        SdW[i] = beta2 * SdW[i] + (1 - beta2) * np.power(dW[i], 2)
        SdB[i] = beta2 * SdB[i] + (1 - beta2) * np.power(dB[i], 2)
        W[i] -= alpha * VdW[i] / np.sqrt(SdW[i] + epsilon)
        B[i] -= alpha * VdB[i] / np.sqrt(SdB[i] + epsilon)

    cache['VdW'] = VdW
    cache['VdB'] = VdB
    cache['SdW'] = SdW
    cache['SdB'] = SdB
    cache['W'] = W
    cache['B'] = B

# Make rough estimation of accuracy on the training records with ages
def estimate_training_set_accuracy(cache):
    A = cache['A']
    L = cache['L']
    Y = cache['Y']
    m = cache['m']

    logger.info('Rough accuracy on training set: ' + str(np.sum((np.abs(np.abs(A[L]) - Y) < 0.5).astype(int)) / float(m)))

main()