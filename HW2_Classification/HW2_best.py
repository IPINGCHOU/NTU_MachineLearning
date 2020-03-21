#%%
# data preprocessing

import numpy as np

np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

#%%

nonbinary_col_idx = np.load('nonbinary_col_idx.npy').tolist()

#%%
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True, specified_column = nonbinary_col_idx)
X_test, _, _= _normalize(X_test, train = False, specified_column = nonbinary_col_idx, X_mean = X_mean, X_std = X_std)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of testing set: {}'.format(test_size))

#%%
# useful function
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

#%%
# cross entropy and gradient
def _cross_entropy_loss(y_pred, Y_label, weight, lamb = 0, pos_weight = 1, punish_weight = 1):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = punish_weight * ((-pos_weight*np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))) + lamb*np.sum(np.square(weight)))
    return cross_entropy

def _gradient(X, Y_label, w, b, lamb = 0, pos_weight =1, punish_weight = 1):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = punish_weight * (pos_weight*Y_label*(1-y_pred) - (1-Y_label)*y_pred)
    w_grad = -np.sum(pred_error * X.T, 1) + 2*lamb*w
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

# def _gradient(X, Y_label, w, b, lamb = 0, pos_weight =1):
#     # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
#     y_pred = _f(X, w, b)
#     pred_error = Y_label- y_pred
#     w_grad = -np.sum(pred_error * X.T, 1) + 2*lamb*w
#     b_grad = -np.sum(pred_error)
#     return w_grad, b_grad

chosen_col = [507, 0, 212, 210, 165, 358, 191, 175, 116, 192, 211, 122, 113, 168, 112, 98, 217, 126, 1, 213, 173, 64, 320, 294, 172, 136, 2, 117, 68, 205, 62, 216, 9, 118, 499, 120, 180, 69, 134, 106, 121, 4, 414, 277, 41, 76, 19, 131, 500, 498, 140, 177, 312, 194, 316, 371, 193, 157, 42, 7, 25, 218, 141, 138, 161, 190, 34, 156, 195, 80, 91, 27, 3, 151, 47, 13, 81, 114, 207, 54, 86, 11, 39, 33, 99, 204, 327, 45, 58, 352, 394, 18, 24, 337, 97, 55, 32, 71, 162, 214, 23, 354, 144, 509, 155, 139, 43, 441, 346, 142, 12, 321, 31, 29, 70, 159, 283, 486, 331, 350, 355, 65, 130, 203, 152, 132, 364, 95, 508, 351, 178, 148, 77, 169, 146, 397, 444, 150, 340, 147, 215, 51, 20, 170, 429, 21, 36, 163, 185, 84, 79, 103, 457, 184, 418, 124, 67, 503, 35, 100, 78, 375, 438, 317, 189, 410, 49, 258, 75, 497, 14, 297, 87, 385, 415, 474, 496, 90, 40, 466, 143, 63, 206, 5, 115, 107, 504, 311, 104, 318, 208, 57, 167, 372, 105, 166, 281, 421, 303, 235, 342, 379, 430, 406, 396, 440, 234, 171, 82, 72, 199, 26, 254, 222, 153, 92, 50, 66, 481, 495, 227, 245, 74, 61, 52, 333, 101, 183, 484, 188, 423, 16, 240, 125, 44, 154, 137, 324, 158, 353, 197, 15, 247, 187, 494, 179, 252, 220, 102, 149, 135, 449, 109, 298, 366, 255, 357, 472, 338, 501, 219, 83, 356, 176, 409, 221, 196, 48, 233, 253, 249, 275, 133, 387, 347, 96, 328, 224, 458, 461, 424, 73, 209, 386, 329, 437, 433, 408, 335, 256, 53, 56, 395, 269, 305, 480, 326, 391, 506, 59, 376, 447, 38, 28, 476, 492, 127, 280, 325, 119, 367, 246, 383, 389, 452, 451, 129, 467, 202, 244, 436, 443, 181, 425, 378, 287, 341, 348, 399, 465, 490, 198, 89, 377, 93, 392, 427, 393, 453, 110, 487, 432, 454, 336, 231, 309, 264, 491, 380, 263, 479, 448, 201, 332, 502, 462, 200, 345, 242, 463, 493, 267, 403, 470, 483, 388, 111, 272, 404, 186, 37, 236, 419, 422, 30, 475, 365, 223, 420, 369, 330, 416, 381, 413, 435, 344, 400, 239, 250, 251, 349, 477, 260, 439, 230, 261, 315, 446, 257, 8, 362, 237, 241, 295, 238, 262, 460, 431, 489, 402, 164, 85, 405, 428, 94, 128, 232, 282, 226]
data_dim = len(chosen_col)
X_train = X_train[:,chosen_col]

#%%
# Training step start
# Zero initialization for weights ans bias

# w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 10000
batch_size = 20000
learning_rate = 0.0005
verbose = 200
# regularization
lamb = 100
#pos_weight = (len(Y_train)-np.sum(Y_train)) / np.sum(Y_train)
pos_weight = 1
punish_weight = 3

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b, lamb, pos_weight, punish_weight)
            
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train, w, lamb, pos_weight, punish_weight) / train_size)

    if epoch % verbose == 0:
        print('======================================')
        print('Now epoch: ' + str(epoch))
        print('Train loss: ' + str(train_loss[-1]))
        print('Train  acc: ' + str(train_acc[-1]))

print('======================================')
print('done')
print('Training     loss: {}'.format(train_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))

# plotting result
#%%
import matplotlib.pyplot as plt

# Loss curve

plt.plot(train_loss)
plt.title('Loss')
plt.legend(['train'])
# plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.title('Accuracy')
plt.legend(['train'])
# plt.savefig('acc.png')
plt.show()

#%%
# output predictions
# Predict testing labels
X_test = X_test[:,chosen_col]
predictions = _predict(X_test, w, b)

output_filename = 'logistic_posw1_pw3_chosencol419'

with open(output_fpath.format(output_filename), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])

#%%