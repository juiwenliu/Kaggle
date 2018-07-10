import copy
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

def main():
    """To run profiler (http://stefaanlippens.net/python_profiling_with_pstats_interactive_mode/):

    1. Launch Command Prompt
    2. Change Directory to cd c:\\Users\\Public\\Documents\\Projects\\Ml\\Kaggle\\Titanic\\Titanic
    3. Run python -m cProfile -o c:\\Users\\Public\\Documents\\Projects\\Ml\\Kaggle\\Titanic\\Titanic\\Profiles\\Profile_2018_0625_2107.profile c:\\Users\\Public\\Documents\\Projects\\Ml\\Kaggle\\Titanic\\Titanic\\classifier_nn_deep_numpy.py
    4. Run python -m pstats "c:\\Users\\Public\\Documents\\Projects\\Ml\\Kaggle\\Titanic\\Titanic\\Profiles\\Profile_2018_0625_2028.profile"
    5. In profile stattistics browser:
       - Run sort cumulative
       - Run stats 10
    """
    logger = initialize_system()
    X, Y = preprocess_data()
    cache = initialize_parameters(X, Y, logger)
    adversaryStreakCounter = 0
    favorableStreakCounter = 0
    
    for i in range(cache['iterationsCount']): # Training iterations
        cache['currentIterationNumber'] = i
        make_forward_propagation(cache)
        compute_cost(cache, logger)
        record_progress(cache, logger)

        if(cache['cost'] >= cache['cost_prev']):
            if(adversaryStreakCounter == 0):
                cache['A_grand_roll_back'] = copy.deepcopy(cache['A_prev'])
                cache['B_grand_roll_back'] = copy.deepcopy(cache['B_prev'])
                cache['SdB_grand_roll_back'] = copy.deepcopy(cache['SdB_prev'])
                cache['SdW_grand_roll_back'] = copy.deepcopy(cache['SdW_prev'])
                cache['VdB_grand_roll_back'] = copy.deepcopy(cache['VdB_prev'])
                cache['VdW_grand_roll_back'] = copy.deepcopy(cache['VdW_prev'])
                cache['W_grand_roll_back'] = copy.deepcopy(cache['W_prev'])
                cache['Z_grand_roll_back'] = copy.deepcopy(cache['Z_prev'])

            adversaryStreakCounter += 1
            favorableStreakCounter = 0

            if (adversaryStreakCounter > cache['adversaryStreakLimit']):
                roll_back_parameters_retune_alpha(cache, adversaryStreakCounter)

                if (np.remainder(adversaryStreakCounter, 20) == 0):
                    make_backward_propagation(cache)
            else:
                make_backward_propagation(cache)

            cache['cost'] = copy.deepcopy(cache['cost_prev'])
        else:
            cache['B_optimal'] = copy.deepcopy(cache['B'])
            cache['W_optimal'] = copy.deepcopy(cache['W'])
            cache['alpha'] = cache['alpha'] if favorableStreakCounter > cache['favorableStreakLimit'] else cache['alpha'] * (1 + np.divide(cache['alphaRecover'], adversaryStreakCounter + 1))
            make_backward_propagation(cache)
            adversaryStreakCounter = 0
            favorableStreakCounter += 1

        update_parameters(cache)

        if (adversaryStreakCounter > cache['adversaryStreakLimit'] and np.remainder(adversaryStreakCounter, 20) == 0):
            cache['A_mini_roll_back'] = copy.deepcopy(cache['A'])
            cache['B_mini_roll_back'] = copy.deepcopy(cache['B'])
            cache['SdB_mini_roll_back'] = copy.deepcopy(cache['SdB'])
            cache['SdW_mini_roll_back'] = copy.deepcopy(cache['SdW'])
            cache['VdB_mini_roll_back'] = copy.deepcopy(cache['VdB'])
            cache['VdW_mini_roll_back'] = copy.deepcopy(cache['VdW'])
            cache['W_mini_roll_back'] = copy.deepcopy(cache['W'])
            cache['Z_mini_roll_back'] = copy.deepcopy(cache['Z'])

    estimate_training_set_accuracy(cache, logger)

def initialize_system():
    logName = 'Logs/Log_' + str(datetime.datetime.now().strftime('%Y_%m%d_%H%M_%S%f')) + '.log'
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
    preProcessedData[np.where(pd.isnull(preProcessedData[:, 9])), 9] = 0

    # In Embarked column, letters converted to digits
    preProcessedData[np.where(pd.isnull(preProcessedData[:, 11])), 11] = 2
    preProcessedData[np.where(preProcessedData[:, 11] == 'C'), 11] = 1
    preProcessedData[np.where(preProcessedData[:, 11] == 'Q'), 11] = -1
    preProcessedData[np.where(preProcessedData[:, 11] == 'S'), 11] = 0

    X = preProcessedData[:, [2, 4, 5, 6, 7, 9, 10, 11]].T # Pick selected 8 columns for training (has to drop column 0 because huge numbers (for example, 891) will easily overflow the sigmoid evaluation)
    Y = preProcessedData[:, [1]].T

    return X, Y
    
def initialize_parameters(X, Y, logger):
    np.random.seed(datetime.datetime.now().microsecond)
    m = X.shape[1]
    N = [X.shape[0], 64, 64, 64, Y.shape[0]]
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
        'adversaryStreakLimit': 200,
        'B': B,
        'B_optimal': None,
        'cost': np.power(10, 3), #random huge number
        'cost_prev': np.power(10, 4), #random huge number
        'cost_rolledBack': 0,
        'dB': dB,
        'dW': dW,
        'favorableStreakLimit': 100,
        'iterationsCount': np.power(10, 5) + 1,
        'L': L,
        'm': m,
        'N': N,
        'SdB': SdB,
        'SdW': SdW,
        'VdB': VdB,
        'VdW': VdW,
        'W': W,
        'W_optimal': None,
        'X': X,
        'Y': Y,
        'Z': Z,
        'zAverage': zAverage,
        'zStdDev': zStdDev,
        'alpha': np.power(10., -2), # Learning rate
        'alphaDecay': 1 - np.power(10., -3), # decay rate on Learning Rate for adversary progress
        'alphaRecover': np.power(10., -4), # learning rate ramp-up rate for favorable progress
        'beta1': 0.9, # Hyperparameter for the moment factor used in ADAM optimization
        'beta2': 0.999, # Hyperparameter for the RMS Prop factor used in ADAM optimization
        'epsilon': np.power(10., -8), # Random small constant for axilliary use
        'lambd': 0.1 # L2 regularization
    }
    
    logger.info(cache)
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
    epsilon = cache['epsilon']

    cache['A_prev'] = copy.deepcopy(cache['A'])
    cache['Z_prev'] = copy.deepcopy(cache['Z'])
    
    for i in range(L):
        Z[i+1] = (np.dot(W[i+1], A[i]) + B[i+1]).astype(np.float32) # Z_l = W_l * A_l-1 + B_l
        
        if (i < L - 1): # For all hidden layers (Layer 1 ~ L-1)
            zAverage[i+1] = np.array(np.average(Z[i+1], axis=1)).reshape(Z[i+1].shape[0], 1)
            zStdDev[i+1] = np.array(np.std(Z[i+1], axis=1)).reshape(Z[i+1].shape[0], 1)

            Z[i+1] = np.divide(Z[i+1] - 0.5 * zAverage[i+1], zStdDev[i+1] + epsilon) # Z Normalization to facilitate learning
            A[i+1] = np.maximum(0, Z[i+1]) # Apply ReLU function max(0, Z)
        else: # For output layer (Layer L)
            A[L] = expit(Z[L]) # Apply sigmoid function 1 / (1 + e^-Z). Used to mitigate overflow of np.exp(). See http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html

# Implement the cost function defined per Cross Entropy Cost function. See https://en.wikipedia.org/wiki/Cross_entropy
def compute_cost(cache, logger): # Only compute top layer cost
    A = cache['A']
    L = cache['L']
    m = cache['m']
    W = cache['W']
    Y = cache['Y']
    epsilon = cache['epsilon']
    lambd = cache['lambd']
    regularizationCostL2 = 0
    cache['cost_prev'] = copy.deepcopy(cache['cost'])

    for i in range(L):
        regularizationCostL2 += np.sum(np.square(W[i+1]))

    regularizationCostL2 = np.divide(regularizationCostL2 * lambd, 2 * m)
    firstTerm = np.dot(Y, np.log(A[L] + epsilon).T) # epsilon to avoid log(0) exception.
    secondTerm = np.dot(1-Y, np.log(1 - A[L] + epsilon).T) # epsilon to avoid log(0) exception.
    cost = np.squeeze(-np.divide(firstTerm + secondTerm, m)) + regularizationCostL2 # np.squeeze to turn one cell array into scalar
    cache['cost'] = cost

def record_progress(cache, logger):
    B_optimal = cache['B_optimal']
    cost = cache['cost']
    N = cache['N']
    cost_prev = cache['cost_prev']
    currentIterationNumber = cache['currentIterationNumber']
    L = cache['L']
    W_optimal = cache['W_optimal']
    alpha = cache['alpha']

    if (cost >= cost_prev):
        flag = '          --'
    else:
        flag = '                ++'

    logger.info('#' + str(currentIterationNumber).rjust(8, '0') + ': ' + str(cost).ljust(25, ' ') + ' ' + str(alpha).ljust(25, ' ') + ' ' + str(datetime.datetime.now()) + ' ' + flag)

    if(currentIterationNumber > 0 and np.remainder(currentIterationNumber, 1000) == 0):
        print('#' + str(currentIterationNumber).rjust(8, '0') + ': ' + str(cost).ljust(25, ' ') + ' ' + str(alpha).ljust(25, ' ') + ' ' + str(datetime.datetime.now()) + ' ' + flag)

        # Final W's and B's are needed for prediction
        logger.info('\n\n\n\n\n\n\n\n--------------------Optimal W--------------------')
        for i in range(L):
            for j in range(N[i+1]):
                for k in range(N[i]):
                    logger.info(W_optimal[i+1][j, k])

        logger.info('\n\n\n\n--------------------Optimal B--------------------')
        for i in range(L):
            logger.info(B_optimal[i+1])

def roll_back_parameters_retune_alpha(cache, adversaryStreakCounter):
    if(adversaryStreakCounter == (cache['adversaryStreakLimit'] + 1) or cache['cost'] > 1.1 * cache['cost_prev']):
        cache['A'] = copy.deepcopy(cache['A_grand_roll_back'])
        cache['B'] = copy.deepcopy(cache['B_grand_roll_back'])
        cache['SdB'] = copy.deepcopy(cache['SdB_grand_roll_back'])
        cache['SdW'] = copy.deepcopy(cache['SdW_grand_roll_back'])
        cache['VdB'] = copy.deepcopy(cache['VdB_grand_roll_back'])
        cache['VdW'] = copy.deepcopy(cache['VdW_grand_roll_back'])
        cache['W'] = copy.deepcopy(cache['W_grand_roll_back'])
        cache['Z'] = copy.deepcopy(cache['Z_grand_roll_back'])
    elif(np.remainder(adversaryStreakCounter, 20) == 1):

        cache['A'] = copy.deepcopy(cache['A_mini_roll_back'])
        cache['B'] = copy.deepcopy(cache['B_mini_roll_back'])
        cache['SdB'] = copy.deepcopy(cache['SdB_mini_roll_back'])
        cache['SdW'] = copy.deepcopy(cache['SdW_mini_roll_back'])
        cache['VdB'] = copy.deepcopy(cache['VdB_mini_roll_back'])
        cache['VdW'] = copy.deepcopy(cache['VdW_mini_roll_back'])
        cache['W'] = copy.deepcopy(cache['W_mini_roll_back'])
        cache['Z'] = copy.deepcopy(cache['Z_mini_roll_back'])
    else:
        cache['A'] = copy.deepcopy(cache['A_prev'])
        cache['B'] = copy.deepcopy(cache['B_prev'])
        cache['SdB'] = copy.deepcopy(cache['SdB_prev'])
        cache['SdW'] = copy.deepcopy(cache['SdW_prev'])
        cache['VdB'] = copy.deepcopy(cache['VdB_prev'])
        cache['VdW'] = copy.deepcopy(cache['VdW_prev'])
        cache['W'] = copy.deepcopy(cache['W_prev'])
        cache['Z'] = copy.deepcopy(cache['Z_prev'])

    if(np.divide(np.abs(cache['cost'] - cache['cost_rolledBack']), cache['cost']) < np.power(10., -12)):
        cache['alpha'] = 0.2
    elif(np.remainder(adversaryStreakCounter, 500) == 0):
        cache['alpha'] = np.divide(cache['alpha'], 2)
    elif(np.remainder(adversaryStreakCounter, 100) == 0):
        cache['alpha'] = cache['alpha'] * np.power(cache['alphaDecay'], 100)
    elif(adversaryStreakCounter == 20):
        cache['alpha'] = cache['alpha'] * np.power(cache['alphaDecay'], 5)
    else:
        cache['alpha'] = cache['alpha'] * cache['alphaDecay']

    cache['cost_rolledBack'] = copy.deepcopy(cache['cost'])
    
# Implement the backward propagation for the SIGMOID -> LINEAR -> [LINEAR->RELU] * (L-1) 
# Implemented references:
# 1. https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building%2Byour%2BDeep%2BNeural%2BNetwork%2B-%2BStep%2Bby%2BStep%2Bv3.ipynb
# 2. https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/Programming%20Assignments/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils_v2.py
# 3. https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Regularization.ipynb
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
    lambd = cache['lambd']

    cache['dB_prev'] = copy.deepcopy(cache['dB'])
    cache['dW_prev'] = copy.deepcopy(cache['dW'])

    # Sigmoid backward-prop implementation. Only for output layer
    # dAL = -np.divide(Y, A[L]+epsilon) + np.divide(1-Y, 1-A[L]+epsilon)
    # AL = expit(Z[L])
    # dZ = dAL * AL * (1 - AL) # element-wise multiplication
    dZL = A[L] - Y
    dW[L] = np.divide(np.dot(dZL, A[L-1].T), m)
    dB[L] = np.divide(np.sum(dZL, axis=1, keepdims=True), m)
    grads['dZ' + str(L)] = dZL

    # ReLU backward-prop implementation. For all hidden layers
    for l in reversed(range(1, L)):
        dA = np.dot(W[l+1].T, grads['dZ' + str(l+1)])
        dZ = np.multiply(dA, np.int64(A[l] > 0))
        dW[l] = np.divide(np.dot(dZ, A[l-1].T), m) + np.divide(lambd * W[l], m)
        dB[l] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
        grads['dZ' + str(l)] = dZ

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
    beta1 = cache['beta1']
    beta2 = cache['beta2']
    epsilon = cache['epsilon']

    cache['B_prev'] = copy.deepcopy(B)
    cache['SdB_prev'] = copy.deepcopy(SdB)
    cache['SdW_prev'] = copy.deepcopy(SdW)
    cache['VdB_prev'] = copy.deepcopy(VdB)
    cache['VdW_prev'] = copy.deepcopy(VdW)
    cache['W_prev'] = copy.deepcopy(W)
    
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

# Make rough estimation of accuracy on the training records with ages
def estimate_training_set_accuracy(cache, logger):
    A = cache['A']
    L = cache['L']
    Y = cache['Y']
    m = cache['m']

    logger.info('Rough accuracy on training set: ' + str(np.sum((np.abs(np.abs(A[L]) - Y) < 0.5).astype(int)) / float(m)))

main()