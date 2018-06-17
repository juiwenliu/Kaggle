import copy
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import re

def main():
    preProcessedData = preprocess_data()
    cache = initialize_parameters(preProcessedData)

    for i in range(10000000):
        cache = make_forward_propagation(cache)
        compute_cost(cache)
        cache = make_backward_propagation(cache)
        cache = update_parameters(cache)
        if (np.remainder(i, 100000) == 0):
            print(str(i) + ": Cost = " + str(cache['Cost']))

    # print('w: ' + str(cache['w']))
    # print('b: ' + str(cache['b']))
    predict(cache)

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
    nx = X.shape[0] # Feature count
    ny = 1 # Output layer unit count
    w = np.random.randn(ny, nx) * 0.01 # Randomly initializing neuros. Factor 0.01 is to ensure the starting parameter to be small
    b = np.zeros((ny, 1)) # Initialize bias to zeros
    A = np.zeros((ny, m)) # Initialize output to zeros
    alpha = 0.001
    epsilon = 0.000000001

    cache = {
        'X': X,
        'Y': Y,
        'm': m,
        'nx': nx,
        'ny': ny,
        'w': w,
        'b': b,
        'A': A,
        'alpha': alpha,
        'epsilon': epsilon
    }

    print('X: ' + str(X.shape))
    print('Y: ' + str(Y.shape))
    print('m: ' + str(m))
    print('nx: ' + str(nx))
    print('ny: ' + str(ny))
    print('w: ' + str(w.shape))
    print('b: ' + str(b.shape))
    print('A: ' + str(A.shape))
    return cache

def make_forward_propagation(cache):
    w = cache['w']
    X = cache['X']
    b = cache['b']
    Z = np.dot(w, X) + b

    cache['A'] = np.reciprocal(1 + np.exp(-Z)) # Apply sigmoid function 1 / (1 + e^-Z)
    return cache

def compute_cost(cache):
    m = cache['m']
    A = cache['A']
    Y = cache['Y']
    cache['Cost'] = np.squeeze(-(np.dot(Y, np.log(A + cache['epsilon']).T) + np.dot(1-Y, np.log(1-A + cache['epsilon']).T))) / m

def make_backward_propagation(cache):
    A = cache['A']
    w = cache['w']
    X = cache['X']
    Y = cache['Y']
    m = cache['m']

    dZ = A - Y
    cache['dw'] = np.dot(dZ, X.T) / m
    cache['db'] = np.sum(dZ, axis=1, keepdims=True) / m
    return cache

def update_parameters(cache):
    cache['w'] -= cache['alpha'] * cache['dw']
    cache['b'] -= cache['alpha'] * cache['db']
    return cache

def predict(cache):
    w = np.array([[-1.04715365, -2.63982776, -0.0442185110, -0.374088222, -0.0682812830, 0.00147806903, 0.567604116]])
    b = np.array([[4.84586073]])
    X = cache['X']
    Y = cache['Y']
    m = cache['m']
    # print(w.shape)
    # print(b.shape)

    print(np.sum((np.abs(np.abs(np.reciprocal(1 + np.exp(-(np.dot(w, X) + b)))) - Y) < 0.5).astype(int)) / float(m))

main()

# After 10000000 training iterations:
# w: [[-1.04715365e+00 -2.63982776e+00 -4.42185110e-02 -3.74088222e-01
#   -6.82812830e-02  1.47806903e-03  5.67604116e-01]]
# b: [[4.84586073]]