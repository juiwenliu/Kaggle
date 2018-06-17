import copy
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import re

def main():
    preProcessedData = preprocess_data()
    cache = initialize_parameters(preProcessedData)

    for i in range(100): # Training iterations
        for j in range(1, cache['L']): # All layers iterations
            make_forward_propagation(cache, j)

        compute_cost(cache)
        print(str(i).rjust(20) + ": Cost = " + str(cache['cost'])) # Print progress with Cost

def preprocess_data():
    with open('train.csv','r') as f:
        rawRows = f.read()
        rawRowsLastNameCommaStripped = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRows) # drop comma between lastname and firstname

    preProcessedData = np.genfromtxt(StringIO(rawRowsLastNameCommaStripped), delimiter= ',', dtype='|U64') # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html Cast all cells to U64 string type to avoid structured ndarray (rank 1 array)
    preProcessedData = preProcessedData[1:] # Drop header row
    preProcessedData[:, 4] = (preProcessedData[:, 4] == 'male').astype(int) # In Sex column, male/female converted to 1/0, respectively
    preProcessedData[:, 10] = (preProcessedData[:, 10] != '').astype(int) # In Cabin column, with/without data converted to 1/0, respectively
    preProcessedData = preProcessedData[np.where(preProcessedData[:, 5] != '')] # Drop rows without data in Age column
    preProcessedData = preProcessedData[:, [1, 2, 4, 5, 6, 7, 9, 10]] # Pick selected 8 columns for training (has to drop column 0 because huge numbers (for example, 891) will easily overflow the sigmoid evaluation)
    preProcessedData = preProcessedData.astype(np.float) # Convert all cells to float
    return preProcessedData

def initialize_parameters(preProcessedData):
    X = copy.deepcopy(preProcessedData[:, [1, 2, 3, 4, 5, 6, 7]]).T # Make deep copy to avoid corrupting raw data
    Y = copy.deepcopy(preProcessedData[:, 0]).reshape(X.shape[1], -1).T # Make deep copy to avoid corrupting raw data. Reshape to address rank-1 array issue
    m = X.shape[1] # Training Set count
    N = [
        X.shape[0], # Feature count
        5, # Layer 1 neural unit count
        4, # Layer 2 neural unit count
        1  # Layer 3 neural unit (output) count
    ]
    L = len(N) # neural network layers count
    W = [np.array([])] # empty array serves as dummy item
    B = [np.array([])] # empty array serves as dummy item
    A = [X] # X is regarded as A0
    alpha = 0.001 # Learning rate
    epsilon = 0.000000001 # For divide-by-zero prevention

    for i in range(1, L):
        W.append(np.random.randn(N[i], N[i-1]) * 0.01) # Randomly initializing neuros. Factor 0.01 is to ensure the starting parameter to be small to stay on linear zone (crucial for gradiant decent on sigmoid)
        B.append(np.zeros((N[i], 1)))
        A.append(np.zeros((N[i], m)))

    cache = {
        'X': X,
        'Y': Y,
        'm': m,
        'N': N,
        'L': L,
        'W': W,
        'B': B,
        'A': A,
        'alpha': alpha,
        'epsilon': epsilon
    }

    print('X: ' + str(X.shape))
    print('Y: ' + str(Y.shape))
    print('m: ' + str(m))
    print('N: ' + str(N))
    print('L: ' + str(L))

    for i in range(L):
        print('W' + str(i) + ': ' + str(W[i].shape))
        print('B' + str(i) + ': ' + str(B[i].shape))
        print('A' + str(i) + ': ' + str(A[i].shape))
    
    print('alpha: ' + str(alpha))
    print('epsilon: ' + str(epsilon))
    return cache

def make_forward_propagation(cache, i):
    W = cache['W']
    A = cache['A']
    B = cache['B']
    Z = np.dot(W[i], A[i-1]) + B[i] # Z_l = W_l * A_l-1 + B_l
    L = cache['L']

    if(i == (L - 1)):
        A[i] = np.reciprocal(1 + np.exp(-Z)) # Apply sigmoid function 1 / (1 + e^-Z)
    else:
        A[i] = np.maximum(0, Z) # Apply ReLU function max(0, Z)

    cache['A'] = A

def compute_cost(cache): # Only compute top layer cost for now, since cost evaluation does not affect training, and cost evaluation slightly slow down training
    m = cache['m']
    A = cache['A']
    Y = cache['Y']
    L = cache['L']
    epsilon = cache['epsilon']
    firstTerm  = np.dot(  Y, np.log(  A[L-1] + epsilon).T) # epsilon to avoid log(0) exception.
    secondTerm = np.dot(1-Y, np.log(1-A[L-1] + epsilon).T) # epsilon to avoid log(0) exception
    cost = -(firstTerm + secondTerm) / m # per Cross Entropy Cost function. See https://en.wikipedia.org/wiki/Cross_entropy
    cache['cost'] = np.squeeze(cost) # np.squeeze to turn one cell array into scalar

main()