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
# def _find_not_binary(X):
#     nums_var = np.shape(X_train)[1]
#     print(nums_var)
#     out_col_idx = []
#     for i in range(nums_var):
#         for j in X_train[:,i]:
#             if j != 0 and j != 1:
#                 out_col_idx.append(i)
#                 break
#     return out_col_idx

nonbinary_col_idx = np.load('nonbinary_col_idx.npy').tolist()
# nonbinary_col_idx = _find_not_binary(X_train)
# np.save('nonbinary_col_idx', nonbinary_col_idx)


#%%
# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True, specified_column = nonbinary_col_idx)
X_test, _, _= _normalize(X_test, train = False, specified_column = nonbinary_col_idx, X_mean = X_mean, X_std = X_std)

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
import torch
import torch.nn as nn
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdm

class hw2Dataset(Dataset):
    def __init__(self,data, train):
        self.data = data
        self.train = train
        self.dim = data.shape[1]

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def _collate_fn(self, data):
        if self.train == True:
            data = np.array(data)
            X = data[:,0:self.dim-1]
            Y = data[:,-1]
            return torch.cuda.FloatTensor(X), torch.cuda.FloatTensor(Y) 
        else:
            return torch.cuda.FloatTensor(data)

class hw2_Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(hw2_Model, self).__init__()

        self.input_dim = input_dim
        self.seq = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.seq(x)

def acc_cal(cur_acc_counts, cur_length, predicts, groundTruth, threshold):
    predicts = predicts > threshold
    corrects = torch.sum(groundTruth.type(torch.uint8) == predicts.type(torch.uint8)).data.item()
    cur_acc_counts += corrects
    cur_length += len(predicts)
    return cur_acc_counts, cur_length

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
hw2_model = hw2_Model(input_dim = data_dim)
hw2_model.to(device)
#%%
input_size = X_train.shape[1]
num_epochs = 50
batch_size = 10000
THRESHOLD = 0.5
learning_rate = 0.01
criteria = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(hw2_model.parameters(), lr = learning_rate)

train_dataset = hw2Dataset(np.concatenate((X_train,Y_train.reshape((len(Y_train),1))), axis=  1), train = True)
valid_dataset = hw2Dataset(np.concatenate((X_dev,Y_dev.reshape((len(Y_dev),1))), axis=  1), train = True)
test_dataset = hw2Dataset(X_test, False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           collate_fn=train_dataset._collate_fn,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, 
                                          batch_size=batch_size, 
                                          collate_fn=valid_dataset._collate_fn,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          collate_fn=test_dataset._collate_fn,
                                          shuffle=False)

for i in range(num_epochs):
    print('\n Epoch: ' + str(i))
    hw2_model.train(True)
    epoch_loss = 0
    epoch_acc_counts = 0
    epoch_length = 0
    trange_train = tqdm(enumerate(train_loader), total = len(train_loader), desc = 'Train')
    for j , (X,Y) in trange_train:
        X = X.to(device)
        Y = Y.to(device)

        logits = hw2_model(X)
        Y = Y.view((len(Y),1))
        loss = criteria(logits,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = torch.sigmoid(logits)
        epoch_acc_counts, epoch_length = acc_cal(epoch_acc_counts, epoch_length, predicted, Y, THRESHOLD)
        trange_train.set_postfix(loss = epoch_loss / (j + 1), Acc = epoch_acc_counts/epoch_length)
    
    trange_valid = tqdm(enumerate(valid_loader), total = len(valid_loader), desc = 'Valid')
    hw2_model.train(False)
    valid_epoch_loss = 0
    valid_epoch_acc_counts = 0
    valid_epoch_length = 0
    for m, (X,Y) in trange_valid:
        X = X.to(device)
        Y = Y.to(device)

        logits = hw2_model(X)
        Y = Y.view((len(Y),1))
        loss = criteria(logits,Y)

        valid_epoch_loss += loss.item()
        predicted = torch.sigmoid(logits)
        valid_epoch_acc_counts, valid_epoch_length = acc_cal(valid_epoch_acc_counts, valid_epoch_length, predicted, Y, THRESHOLD)
        trange_valid.set_postfix(loss = valid_epoch_loss / (m + 1), Acc = valid_epoch_acc_counts/valid_epoch_length)
    
# %%
# prediction
ans = []
trange_test = tqdm(enumerate(test_loader), total=len(test_loader), desc = 'Test')
for z, (X) in trange_test:
    X = X.to(device)
    logits = hw2_model(X)
    predicted = logits > THRESHOLD
    predicted = predicted.type(torch.uint8).tolist()
    ans.extend(predicted)

predictions = np.array(ans).reshape(-1)
output_filename = 'NN_first_try'
with open(output_fpath.format(output_filename), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# %%
