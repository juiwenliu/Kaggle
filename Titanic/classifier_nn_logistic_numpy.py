import copy
from io import StringIO
import numpy as np
import re

def main():
    X, Y = preprocess_data()
    # cache = initialize_parameters(X, Y)
    # forward_propagation(cache)
    # print(cache)

def preprocess_data():
    with open('train.csv','r') as f:
        rawRow = f.read()
        rawRowLastNameCommaStripped = re.sub(r'"(.*),(.*)"', r'\1 \2', rawRow) # drop comma between lastname and firstname
        rawArrayLastNameCommaStripped = np.genfromtxt(StringIO(rawRowLastNameCommaStripped), delimiter= ',', dtype='|U64')
        # StringIO() is needed for a non-file-name object, see https://docs.scipy.org/doc/numpy-1.13.0/user/basics.io.genfromtxt.html
        # cast all cells to U64 string type to avoid structured ndarray (rank 1 array)

    rawArrayLastNameCommaStrippedSexBinarized = copy.deepcopy(rawArrayLastNameCommaStripped)
    rawArrayLastNameCommaStrippedSexBinarized[1:, 4] = (rawArrayLastNameCommaStrippedSexBinarized[1:, 4] == 'male').astype(int) # Male/female converted to 1/0, respectively

    rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized = copy.deepcopy(rawArrayLastNameCommaStrippedSexBinarized)
    rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized[1:, 10] = (rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized[1:, 10] != '').astype(int)


    hasNoAge = (rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized[1:, 5] != '').astype(int) - 1
    # print(hasNoAge)

    temp = copy.deepcopy(rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized[1:, [0, 2, 4, 5, 6, 7, 9, 10]]) # Make deep copy (skiping header row) to avoid corrupting raw data
    temp2 = temp[:30]
    temp3 = temp2[np.where(temp2[:, 3] != '')]
    temp3[:, 3] = temp3[:, 3].astype(np.float)

    temp4 = temp3[np.where(temp3[:] == '')]

    # print(temp2.shape)
    print(temp3)
    print(temp4)
    print(temp5)
#     ctr = 0
#     for i in range(temp.shape[0]):
#         if(temp[i, 3] == ''):
#             temp[i, 3]

#     X = np.zeros((temp.shape[0], temp.shape[1]))
    
#     # X[:, 0] = temp[:, 0].astype(np.float)
#     # X[:, 1] = temp[:, 1].astype(np.float)
#     # X[:, 2] = temp[:, 2].astype(np.float)
#     temp[:, 3] = hasNoAge + temp[:, 3]
#     # print(temp[:, 3])
#     # X[:, 3] = temp[:, 3].astype(np.int)
#     # X[:, 4] = temp[:, 4].astype(np.float)
#     # X[:, 5] = temp[:, 5].astype(np.float)
#     # X[:, 6] = temp[:, 6].astype(np.float)
#     # X[:, 7] = temp[:, 7].astype(np.float)

#     # for i in range(temp.shape[1]):
#     #     X = temp[:, i].astype(np.float)

#     # print(X[:, 3])

#     # K = np.float(X)# X.astype(np.float) #float(X)#.astype(np.float)
#     # print(X)
#     Y = copy.deepcopy(rawArrayLastNameCommaStrippedSexBinarizedCabinBinarized[1:, 1]).reshape(X.shape[0], -1).astype(int) # Make deep copy (skiping header row) to avoid corrupting raw data
    X = []
    Y = []

    return X, Y

# def initialize_parameters(X, Y):
#     m = X.shape[0] # Training Set count
#     nx = X.shape[1] # Feature count
#     ny = 1 # Output layer unit count
#     w = np.random.randn(ny, nx) # Randomly initializing neuros
#     b = np.zeros((ny, 1)) # Initialize bias to zeros
#     Z = np.zeros((ny, m)) # Initialize output to zeros

#     print('X: ' + str(X.shape))
#     print('Y: ' + str(Y.shape))
#     print('m: ' + str(m))
#     print('nx: ' + str(nx))
#     print('ny: ' + str(ny))
#     print('w: ' + str(w.shape))
#     print('b: ' + str(b.shape))
#     print('Z: ' + str(b.shape))

#     cache = {
#         'X': X,
#         'Y': Y,
#         'm': m,
#         'nx': nx,
#         'ny': ny,
#         'w': w,
#         'b': b,
#         'Z': Z
#     }

#     return cache

# def forward_propagation(cache):
#     Z = cache['Z']
#     w = cache['w']
#     X = cache['X']
#     b = cache['b']
#     Z = np.dot(w, X) + b
#     print('Z: ' + str(Z))

main()
