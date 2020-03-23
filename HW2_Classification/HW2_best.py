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

nonbinary_col_idx = np.load('nonbinary_col_idx.npy').tolist()
X_mean, X_std = np.load('X_mean_nonbinary.npy'), np.load('X_std_nonbinary.npy')
X_test, _, _= _normalize(X_test, train = False, specified_column = nonbinary_col_idx, X_mean = X_mean, X_std = X_std)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
hw2_model = hw2_Model(input_dim = X_test.shape[1])
hw2_model.to(device)
hw2_model.load_state_dict(torch.load('NN_first_try'))
hw2_model.train(False)

test_dataset = hw2Dataset(X_test, False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=500, 
                                          collate_fn=test_dataset._collate_fn,
                                          shuffle=False)

ans = []
trange_test = tqdm(enumerate(test_loader), total=len(test_loader), desc = 'Test')
for z, (X) in trange_test:
    X = X.to(device)
    logits = hw2_model(X)
    predicted = logits > 0.5
    predicted = predicted.type(torch.uint8).tolist()
    ans.extend(predicted)

predictions = np.array(ans).reshape(-1)
with open(output_route, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))