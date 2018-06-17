import copy
from io import StringIO
import numpy as np
import re

def main():
    preProcessedData = preprocess_data()
    cache = initialize_parameters(preProcessedData)
    # forward_propagation(cache)
    # print(cache)

def preprocess_data():
    with open('train.csv','r') as f:
        rawRows = f.read()
        rawRowsLastNameCommaStripped = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRows) # drop comma between lastname and firstname

    preProcessedData = np.genfromtxt(StringIO(rawRowsLastNameCommaStripped), delimiter= ',', dtype='|U64')
    # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html
    # Cast all cells to U64 string type to avoid structured ndarray (rank 1 array)

    # The following preprocessings start at row 1 to avoid header row
    preProcessedData[1:, 4] = (preProcessedData[1:, 4] == 'male').astype(int) # In Sex column, male/female converted to 1/0, respectively
    preProcessedData[1:, 10] = (preProcessedData[1:, 10] != '').astype(int) # In Cabin column, with/without data converted to 1/0, respectively
    preProcessedData = preProcessedData[np.where(preProcessedData[1:, 5] != '')] # Drop rows without data in Age column

    return preProcessedData

def initialize_parameters(preProcessedData):
    X = copy.deepcopy(preProcessedData[1:, [0, 2, 4, 5, 6, 7, 9, 10]]) # Make deep copy (skiping header row) to avoid corrupting raw data
    Y = copy.deepcopy(preProcessedData[1:, 1]).reshape(X.shape[0], -1) # Make deep copy (skiping header row) to avoid corrupting raw data. Reshape to fix rank-1 array issue
    m = X.shape[0] # Training Set count
    nx = X.shape[1] # Feature count
    ny = 1 # Output layer unit count
    w = np.random.randn(ny, nx) # Randomly initializing neuros
    b = np.zeros((ny, 1)) # Initialize bias to zeros
    Z = np.zeros((ny, m)) # Initialize output to zeros

    print('X: ' + str(X.shape))
    print('Y: ' + str(Y.shape))
    print('m: ' + str(m))
    print('nx: ' + str(nx))
    print('ny: ' + str(ny))
    print('w: ' + str(w.shape))
    print('b: ' + str(b.shape))
    print('Z: ' + str(b.shape))

    cache = {
        'X': X,
        'Y': Y,
        'm': m,
        'nx': nx,
        'ny': ny,
        'w': w,
        'b': b,
        'Z': Z
    }

    return cache

# def forward_propagation(cache):
#     Z = cache['Z']
#     w = cache['w']
#     X = cache['X']
#     b = cache['b']
#     Z = np.dot(w, X) + b
#     print('Z: ' + str(Z))

main()
