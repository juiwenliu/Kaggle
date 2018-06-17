import copy
from io import StringIO
import numpy as np
import re

def main():
    preProcessedData = preprocess_data()
    cache = initialize_parameters(preProcessedData)
    forward_propagation(cache)

def preprocess_data():
    with open('train.csv','r') as f:
        rawRows = f.read()
        rawRowsLastNameCommaStripped = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRows) # drop comma between lastname and firstname

    preProcessedData = np.genfromtxt(StringIO(rawRowsLastNameCommaStripped), delimiter= ',', dtype='|U64') # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html Cast all cells to U64 string type to avoid structured ndarray (rank 1 array)
    preProcessedData = preProcessedData[1:] # Drop header row
    preProcessedData[:, 4] = (preProcessedData[:, 4] == 'male').astype(int) # In Sex column, male/female converted to 1/0, respectively
    preProcessedData[:, 10] = (preProcessedData[:, 10] != '').astype(int) # In Cabin column, with/without data converted to 1/0, respectively
    preProcessedData = preProcessedData[np.where(preProcessedData[:, 5] != '')] # Drop rows without data in Age column
    preProcessedData = preProcessedData[:, [0, 1, 2, 4, 5, 6, 7, 9, 10]] # Pick selected 8 columns for training
    preProcessedData = preProcessedData.astype(np.float) # Convert all cells to float
    return preProcessedData

def initialize_parameters(preProcessedData):
    X = copy.deepcopy(preProcessedData[:, [0, 2, 3, 4, 5, 6, 7, 8]]).T # Make deep copy to avoid corrupting raw data
    Y = copy.deepcopy(preProcessedData[:, 1]).reshape(X.shape[1], -1).T # Make deep copy to avoid corrupting raw data. Reshape to address rank-1 array issue
    m = X.shape[1] # Training Set count
    nx = X.shape[0] # Feature count
    ny = 1 # Output layer unit count
    w = np.random.randn(ny, nx) * 0.01 # Randomly initializing neuros. Factor 0.01 is to ensure the starting parameter to be small
    b = np.zeros((ny, 1)) # Initialize bias to zeros
    Z = np.zeros((ny, m)) # Initialize output to zeros

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

    print('X: ' + str(X.shape))
    print('Y: ' + str(Y.shape))
    print('m: ' + str(m))
    print('nx: ' + str(nx))
    print('ny: ' + str(ny))
    print('w: ' + str(w.shape))
    print('b: ' + str(b.shape))
    print('Z: ' + str(Z.shape))

    return cache

def forward_propagation(cache):
    w = cache['w']
    X = cache['X']
    b = cache['b']
    cache['Z'] = np.dot(w, X) + b
    return cache

main()
