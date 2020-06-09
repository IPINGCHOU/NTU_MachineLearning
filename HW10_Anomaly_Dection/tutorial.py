#%%
# read data
import numpy as np
import os
import random
import torch
import seaborn as sns

train = np.load('train.npy')
test = np.load('test.npy')

PRED_FOLDER = './prediction/'
SEED = 1234

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
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans

task = 'knn'
if task == 'knn':
    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    scores = list()
    for n in range(1, 10):
    #   kmeans_x = MiniBatchKMeans(n_clusters=n, batch_size=100, random_state=SEED).fit(x)
      kmeans_x = KMeans(n_clusters=n, random_state=SEED).fit(x)
      y_cluster = kmeans_x.predict(y)
      y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - y), axis=1)

      y_pred = y_dist

    #   score = f1_score(y_label, y_pred, average='micro')
    #   score = roc_auc_score(y_label, y_pred, average='micro')
    #   scores.append(score)
    # print(np.max(scores), np.argmax(scores))
    # print(scores)
    # print('auc score: {}'.format(np.max(scores)))
#%%
with open(PRED_FOLDER + 'tutorial_fullknn.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
print('knn saved')

#%%

from sklearn.decomposition import PCA

task = 'pca'
if task == 'pca':

    x = train.reshape(len(train), -1)
    y = test.reshape(len(test), -1)
    pca = PCA(n_components=2, random_state=SEED).fit(x)

    y_projected = pca.transform(y)
    y_reconstructed = pca.inverse_transform(y_projected)  
    dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))
    
    y_pred = dist

with open(PRED_FOLDER + 'tutorial_pca.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
print('pca saved')
#%%
# kernel pca
from sklearn.decomposition import PCA, KernelPCA

x = train.reshape(len(train), -1)
y = test.reshape(len(test), -1)
transformer = KernelPCA(n_components=2, kernel='rbf', n_jobs=-1, fit_inverse_transform=True,random_state=SEED)
pca = transformer.fit(x)

y_projected = pca.transform(y)
y_reconstructed = pca.inverse_transform(y_projected)  
dist = np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1)

y_pred = dist
with open(PRED_FOLDER + 'tutorial_kernelPCA_nosqrt.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
print('pca saved')

#%%
import torch
from torch import nn
import torch.nn.functional as F


class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3)
        )


        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
		    nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return mse + KLD

#%%
# run model
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

task = 'ae'
#{'fcn', 'cnn', 'vae'} 
model_type = 'vae'
if task == 'ae':
    num_epochs = 1000
    batch_size = 1024
    learning_rate = 1e-3

    #{'fcn', 'cnn', 'vae'} 
    # model_type = 'cnn'

    x = train
    if model_type == 'fcn' or model_type == 'vae':
        x = x.reshape(len(x), -1)
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate)
    
    best_loss = np.inf
    total_loss = 0
    model.train()
    step = 0
    for epoch in range(num_epochs):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        # ===================save====================
        if total_loss/step < best_loss:
            best_loss = total_loss/step
            torch.save(model, 'best_model_{}.pt'.format(model_type))
            print('\n epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, total_loss/step), end = '\n')

        # ===================log========================
        print('\r epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, total_loss/step), end = ' ')
        total_loss, step = 0,0

#%%
# eval
task = 'ae'
if task == 'ae':
    if model_type == 'fcn' or model_type == 'vae':
        y = test.reshape(len(test), -1)
    else:
        y = test
        
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')

    model.eval()
    reconstructed = list()
    for i, data in enumerate(test_dataloader): 
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
    y_pred = anomality

    with open(PRED_FOLDER + 'prediction_'+str(model_type)+'.csv', 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
# %%
