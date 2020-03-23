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
# find the cols that are not binary input
def _find_not_binary(X):
    nums_var = np.shape(X_train)[1]
    print(nums_var)
    out_col_idx = []
    for i in range(nums_var):
        for j in X_train[:,i]:
            if j != 0 and j != 1:
                out_col_idx.append(i)
                break
    return out_col_idx

nonbinary_col_idx = np.load('nonbinary_col_idx.npy').tolist()
# nonbinary_col_idx = _find_not_binary(X_train)
# np.save('nonbinary_col_idx', nonbinary_col_idx)


#%%
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True, specified_column = None)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

np.save('X_mean', X_mean)
np.save('X_std', X_std)
#%%

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

#%%
# feature importance
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators = 200,
#                            n_jobs = -1,
#                            max_features = None,
#                            min_samples_leaf = 50,
#                            max_depth = 100,
#                            oob_score = True,
#                            bootstrap = True,
#                            random_state = 42)
# rf.fit(X_train, Y_train)
# print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, Y_train), 
#                                                                                              rf.oob_score_,
#                                                                                              rf.score(X_dev, Y_dev)))

#%%
# from sklearn.metrics import r2_score
# from rfpimp import permutation_importances

# def r2(rf, X_train, y_train):
#     return r2_score(y_train, rf.predict(X_train))

# import pandas as pd
# perm_imp_rfpimp = permutation_importances(rf, pd.DataFrame(X_train), pd.DataFrame(Y_train), r2)

# import seaborn as sns
# ax = sns.barplot(y = 'Importance', x = perm_imp_rfpimp.index, data=perm_imp_rfpimp) 
# #%%
# # chosen one
# first_col = 42
# back_col = 499
# chosen_col = perm_imp_rfpimp.index[0:first_col].tolist()
# chosen_col.extend(perm_imp_rfpimp.index[back_col:])
# data_dim = len(chosen_col)
# print(chosen_col)

# X_train = X_train[:,chosen_col]
# X_dev = X_dev[:,chosen_col]

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

#%%
# Training step start
# normal initialization for weights ans bias
print(X_train.shape)
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 2000
batch_size = 5000
learning_rate = 0.01
verbose = 100
# regularization
# lamb = 1
pos_weight = 1
# pos_weight = (len(Y_train)-np.sum(Y_train))/np.sum(Y_train)
punish_weight = 1

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calcuate the number of parameter updates
step = 1


lamb_idx = [0]
lamb_train_loss, lamb_valid_loss = [],[]
lamb_train_acc, lamb_valid_acc = [], []

# Iterative training
for lamb in lamb_idx:
    print('======================================')
    print('Now testing lambda: ' + str(lamb))
    print('======================================')
    # initialize 
    step = 1
    w = np.zeros((data_dim,)) 
    b = np.zeros((1,))

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

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev, w, lamb, pos_weight, punish_weight) / dev_size)

        if epoch % verbose == 0:
            print('======================================')
            print('Now epoch: ' + str(epoch))
            print('Now lr   : ' + str(learning_rate/np.sqrt(step)))
            print('Train loss: ' + str(train_loss[-1]))
            print('Dev   loss: ' + str(dev_loss[-1]))
            print('Train  acc: ' + str(train_acc[-1]))
            print('Dev    acc: ' + str(dev_acc[-1]))

    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))
    lamb_train_loss.append(train_loss[-1])
    lamb_valid_loss.append(dev_loss[-1])
    lamb_train_acc.append(train_acc[-1])
    lamb_valid_acc.append(dev_acc[-1])

#%%
# plotting result
# plotting different lambda result
import matplotlib.pyplot as plt
plt.plot(lamb_train_loss)
plt.plot(lamb_valid_loss)
plt.title('Loss of train and valid with differnet lambda')
plt.legend(['train','valid'])
plt.xticks(range(0,5,1), lamb_idx)
plt.ylabel('Loss')
plt.xlabel('Lambda')
plt.show()

plt.plot(lamb_train_acc)
plt.plot(lamb_valid_acc)
plt.title('Acc of train and valid with differnet lambda')
plt.legend(['train','valid'])
plt.xticks(range(0,5,1), lamb_idx)
plt.ylabel('Loss')
plt.xlabel('Lambda')
plt.show()

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
# plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
# plt.savefig('acc.png')
plt.show()

#%%
# output predictions
# Predict testing labels
predictions = _predict(X_test, w, b)

output_filename = 'logistic_PCA200'

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