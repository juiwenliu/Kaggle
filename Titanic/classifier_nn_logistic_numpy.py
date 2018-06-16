import copy
from io import StringIO
import numpy as np
import re

def main():
    X, Y = preprocess_data()
    cache = initialize_parameters(X, Y)
    print(cache)

def preprocess_data():
    with open('train.csv','r') as f:
        rawRow = f.read()
        rawRowWithoutLastNameComma = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRow) # drop comma between lastname and firstname
        rawArrayWithoutLastNameComma = np.genfromtxt(StringIO(rawRowWithoutLastNameComma), delimiter= ',', dtype='|U64') 
        # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html
        # cast all cells to U64 string type to avoid structured ndarray (rank 1 array)

    X = copy.deepcopy(rawArrayWithoutLastNameComma[1:, [0, 2, 4, 5, 6, 7, 9, 10]]) # Make deep copy (skiping header row) to avoid corrupting raw data
    Y = copy.deepcopy(rawArrayWithoutLastNameComma[1:, 1]).reshape(X.shape[0], -1) # Make deep copy (skiping header row) to avoid corrupting raw data

    return X, Y

def initialize_parameters(X, Y):
    cache = {}
    m = X.shape[0] # Training Set count
    nx = X.shape[1] # Feature count
    W = np.random.randn(1, nx)

    print('X: ' + str(X.shape))
    print('Y: ' + str(Y.shape))
    print('m: ' + str(m))
    print('nx: ' + str(nx))
    print('W: ' + str(W.shape))

    cache = {
        'X': X,
        'Y': Y,
        'm': m,
        'nx': nx,
        'W': W
    }

    return cache

main()
