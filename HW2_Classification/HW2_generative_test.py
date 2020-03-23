import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

input_route = sys.argv[5]
output_route = sys.argv[6]

with open(input_route) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)

nonbinary_col_idx = np.load('nonbinary_col_idx.npy').tolist()
X_mean, X_std = np.load('X_mean_nonbinary.npy'), np.load('X_std_nonbinary.npy')
X_test, _, _= _normalize(X_test, train = False, specified_column = nonbinary_col_idx, X_mean = X_mean, X_std = X_std)
w,b = np.load('w_generative.npy'), np.load('b_generative.npy')

predictions = 1 - _predict(X_test, w, b)

with open(output_route, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
