#%%
# read data
import numpy as np
import os
import random
import torch
train = np.load('train.npy')
test = np.load('test.npy')

PRED_FOLDER = './prediction/'
SEED = 1234

LATENT_SIZE = 20

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(SEED)

#%%
import torch
from torch import nn
import torch.nn.functional as F

#%%
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    ) 
#%%
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        # U-left
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # U-right
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1, bias=False)

        # latent vae
        self.fc1 = nn.Linear(512*4*4, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc21 = nn.Linear(512, LATENT_SIZE)
        self.fc22 = nn.Linear(512, LATENT_SIZE)

        self.fc3 = nn.Linear(LATENT_SIZE, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512*4*4)
        self.bn4 = nn.BatchNorm1d(512*4*4)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.normal(0,1, size = mu.size()).cuda()
        # esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def forward(self, x):
        # U left
        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1) # 16 16 64

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2) # 8 8 128
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) # 4 4 256
        
        x = self.dconv_down4(x) 
        # x = self.maxpool(x) # 4 4 512
        
        # vae part
        # flatten
        b, k, w, s = x.shape
        x = x.view(b, k*w*s)
        x = self.relu(self.bn1(self.fc1(x)))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        z = self.reparametrize(mu, logvar)
        
        x = self.relu(self.bn3(self.fc3(z)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = x.view(b,k,w,s)

        # U-right
        x = self.upsample(x) # 4 4 256
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)  # 8 8 128
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x) # 16 16 64
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x) # 32 32 3
        out = self.tanh(out)

        return z, out, mu, logvar

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # KL divergence
    return mse + KLD
#%%
# training
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

NUM_EPOCHS = 100
BATCH_SIZE = 512
LR = 5e-05
DEVICE = 'cuda:0'
MODEL_TYPE = 'unet_vae'

data = torch.tensor(train, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

model = AE().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

best_loss = np.inf
total_loss = 0
model.train()
step = 0

for epoch in range(NUM_EPOCHS):
    for data in train_dataloader:
        img = data[0].transpose(3, 1).to(DEVICE)
    
        enc, dec, mu, logvar = model(img)
        loss = loss_vae(dec,img,mu,logvar,criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1


    average_loss = total_loss / step
    print('\n unet epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, NUM_EPOCHS, average_loss), end = '\n')
    
    if average_loss < best_loss:
        bets_loss = average_loss
        torch.save(model, 'best_model_{}.pt'.format(MODEL_TYPE))
        print('Model Saved!!!')
        
    # print('\r unet epoch [{}/{}], loss:{:.4f}'
    #           .format(epoch + 1, NUM_EPOCHS, average_loss), end = ' ')
    total_loss, step = 0,0


#%%
# test

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

DEVICE = 'cuda:0'

batch_size = 256
def get_latents(data):
    data = torch.tensor(data, dtype=torch.float)
    dataset = TensorDataset(data)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)

    model = torch.load('best_model_{}.pt'.format(MODEL_TYPE), map_location='cuda')
    model.eval()
    latent = []

    for i, data in enumerate(dataloader):
        img = data[0].transpose(3, 1).cuda()
        enc, output, _, _ = model(img)
        if i == 0:
            latents = enc.view(output.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, enc.view(output.size()[0], -1).cpu().detach().numpy()), axis = 0)

    print('Latents Shape:', latents.shape)
    return latents

def get_reconstruct(data):
    data = torch.tensor(data, dtype=torch.float)
    dataset = TensorDataset(data)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)

    model = torch.load('best_model_{}.pt'.format(MODEL_TYPE), map_location='cuda')
    model.eval()

    reconstruct = []
    for i, data in enumerate(dataloader):
        img = data[0].transpose(3,1).cuda()
        enc, output, _, _ = model(img)
        output = output.transpose(3,1)
        reconstruct.append(output.cpu().detach().numpy())

    reconstruct = np.concatenate(reconstruct, axis = 0)
    print('Reconstruce shape: ', reconstruct.shape)
    return reconstruct

def pred_save(pred, name):
    with open(name, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(pred)):
            f.write('{},{}\n'.format(i+1, pred[i]))

#%%
test_latents = get_latents(test)
train_reconstruct = get_reconstruct(train)
test_reconstruct = get_reconstruct(test)
#%%
import matplotlib.pyplot as plt
def show_reconstruct(pic, flag):
    fig, axs = plt.subplots(1,2)
    if flag == 'test':
        axs[0].imshow(test[pic])
        axs[1].imshow(test_reconstruct[pic])
    else:
        axs[0].imshow(train[pic])
        axs[1].imshow(train_reconstruct[pic])

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
#%%
for i in range(40,60,1):
    show_reconstruct(i, 'test')

for i in range(20,40,1):
    show_reconstruct(i, 'train')
#%%
# reconstruct 
import seaborn as sns
# train
train_anomality = np.sqrt(np.sum(np.square(train_reconstruct - train).reshape(len(train),-1), axis = 1))
sns.distplot(train_anomality)

anomality = np.sqrt(np.sum(np.square(test_reconstruct - test).reshape(len(test),-1), axis = 1))
sns.distplot(anomality)
pred_save(anomality, PRED_FOLDER + 'prediction_unet_vae.csv')
# pred = anomality

# PRED_FOLDER + 'prediction_'+str('unet')+'.csv'

# %%

from sklearn.decomposition import PCA, KernelPCA

print('PCA fitting...')
transformer = PCA(n_components=2, random_state=SEED)
y_projected = transformer.fit_transform(test_latents)
y_reconstructed = transformer.inverse_transform(y_projected)  
dist = np.sqrt(np.sum(np.square(y_reconstructed - test_latents).reshape(len(test_latents), -1), axis=1))
sns.distplot(dist)
pred_save(dist, PRED_FOLDER + 'prediction_unet_vae_pca_reconstruced.csv')

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
print('TSNE fitting...')
tsne = TSNE(n_components=2, random_state=SEED, verbose = True)
y_TSNE = tsne.fit_transform(test_latents)
plt.scatter(y_TSNE[:,0], y_TSNE[:,1], s = 1)

rmse_tsne_test = np.sqrt(np.square(y_TSNE[:,0] - np.mean(y_TSNE[:,0])) + np.square(y_TSNE[:,1] - np.mean(y_TSNE[:,1])))
sns.distplot(rmse_tsne_test)
pred_save(rmse_tsne_test, PRED_FOLDER + 'prediction_unet_vae_tsne_rmse.csv')
# %%
from sklearn.neighbors import KernelDensity
kd = KernelDensity()
kd.fit(test_latents)
score = [kd.score(i.reshape(1,-1)) for i in test_latents]
score = score - np.min(score)
sns.distplot(score)
pred_save(score, PRED_FOLDER + 'prediction_unet_vae_latentkd.csv')

# %%
