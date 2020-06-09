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
import torch
from torch import nn
import torch.nn.functional as F


class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True), 
            nn.Linear(256, 128), 
            nn.ReLU(True),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True), 
            nn.Linear(512, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec

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

        self.bn1 = nn.Linear(48*4*4, 512)
        self.bn2 = nn.Linear(512, 384)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, 4, stride=2, padding=1),  # [batch, 6, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        b, k, w, h = enc.shape

        enc_b1 = F.relu(self.bn1(enc.view(b, k*w*h)))
        enc_b2 = F.relu(self.bn2(enc_b1))


        dec = self.decoder(enc_b2.view(b, 24, 4, 4))
        return enc, dec, (enc_b1, enc_b2)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc31 = nn.Linear(400, 32)
        self.fc32 = nn.Linear(400, 32)
        self.fc4 = nn.Linear(32, 400)
        self.fc5 = nn.Linear(400,800)
        self.fc6 = nn.Linear(800, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.normal(0,1, size = mu.size()).cuda()
        # esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return F.tanh(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, self.decode(z), mu, logvar, 

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
		    nn.Conv2d(64, 128, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )

        # self.fc1 = nn.Linear(48*4*4, 384)
        # self.fc2 = nn.Linear(384, 192)
        self.fc31 = nn.Linear(128*4*4, 500)
        self.fc32 = nn.Linear(128*4*4, 500)
        # self.fc4 = nn.Linear(32, 384)
        # self.fc5 = nn.Linear(192, 384)
        self.fc6 = nn.Linear(800, 128*4*4)
    
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        # reshape
        b, k, w, s = x.shape
        x = x.view(b, k*w*s)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc31(x), self.fc32(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.normal(0,1, size = mu.size()).cuda()
        # esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def decode(self, z):
        # z = F.relu(self.fc4(z))
        # z = F.relu(self.fc5(z))
        z = self.fc6(z)
        # reshape
        b = len(z)
        z = z.view(b, 128, 4 ,4)
        z = self.decoder_conv(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, self.decode(z), mu, logvar

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
# run model
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

task = 'ae'
#{'fcn', 'cnn', 'vae'} 
model_type = 'vae_cnn'
model_specify = 'enc800'
if task == 'ae':
    num_epochs = 1000
    batch_size = 512
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


    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), 'vae_cnn':CVAE()}
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
            if model_type == 'cnn' or model_type == 'vae_cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            if model_type == 'vae' or model_type == 'vae_cnn':
                enc, dec, mu, logvar  = model(img)
            elif model_type == 'fcn':
                enc, dec = model(img)
            elif model_type == 'cnn':
                enc, dec, bcs = model(img)

            if model_type == 'vae' or model_type == 'vae_cnn':
                loss = loss_vae(dec, img, mu, logvar, criterion)
            else:
                loss = criterion(dec, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        # ===================save====================
        if total_loss/step < best_loss:
            best_loss = total_loss/step
            torch.save(model, 'best_model_{}_{}.pt'.format(model_type, model_specify))
            print('\n epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, total_loss/step), end = '\n')

        # ===================log========================
        print('\r epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, total_loss/step), end = ' ')
        total_loss, step = 0,0

#%%
# eval
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

batch_size = 2048

def get_latents(data, bottle_neck = False):
    if model_type =='fcn' or model_type == 'vae':
        data = data.reshape(len(data), -1)
    data = torch.tensor(data, dtype=torch.float)
    dataset = TensorDataset(data)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model = torch.load('best_model_{}_{}.pt'.format(model_type, model_specify), map_location='cuda')
    # model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')
    print(model)
    model.eval()
    latents = []

    for i, data in enumerate(dataloader):
        if model_type=='cnn' or model_type=='vae_cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        
        if model_type == 'vae' or model_type == 'vae_cnn':
            enc, dec, mu, logvar  = model(img)
        elif model_type == 'fcn':
            enc, dec = model(img)
        elif model_type == 'cnn':
            enc, dec, bcs = model(img)

        if bottle_neck != True:
            if i == 0:
                latents = enc.view(dec.size()[0], -1).cpu().detach().numpy()
            else:
                latents = np.concatenate((latents, enc.view(dec.size()[0], -1).cpu().detach().numpy()), axis = 0)
        else:
            if i == 0:
                latents = bcs[1].cpu().detach().numpy()
            else:
                latents = np.concatenate((latents, bcs[1].cpu().detach().numpy()), axis = 0)

    print('Latents Shape:', latents.shape)
    return latents
#%%
def get_reconstruct(data):
    if model_type =='fcn' or model_type == 'vae':
        data = data.reshape(len(data), -1)
    data = torch.tensor(data, dtype=torch.float)
    dataset = TensorDataset(data)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model = torch.load('best_model_{}_{}.pt'.format(model_type, model_specify), map_location='cuda')
    # model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')
    print(model)
    model.eval()

    reconstruct = []
    for i, data in enumerate(dataloader):
        if model_type=='cnn' or model_type == 'vae_cnn':
            img = data[0].transpose(3,1).cuda()
        else:
            img = data[0].cuda()

        if model_type == 'vae' or model_type == 'vae_cnn':
            enc, dec, mu, logvar  = model(img)
        elif model_type == 'fcn':
            enc, dec = model(img)
        elif model_type == 'cnn':
            enc, dec, bcs = model(img)

        if model_type=='cnn' or model_type == 'vae_cnn':
            dec = dec.transpose(3,1)
        elif model_type == 'fcn' or model_type == 'vae':
            b ,_ =dec.shape
            dec = dec.view(b, 32,32,3)

        reconstruct.append(dec.cpu().detach().numpy())

    reconstruct = np.concatenate(reconstruct, axis = 0)
    return reconstruct

#%%
# reconstruct 
model_type = 'vae_cnn'
model_specify = 'enc800'
test_reconstruct = get_reconstruct(test)
train_reconstruct = get_reconstruct(train)
if model_type=='fcn' or model_type == 'vae':
    test_temp = test.reshape(len(test),-1)
    anomality = np.sqrt(np.sum(np.square(test_reconstruct - test).reshape(len(test),-1), axis = 1))
else:
    anomality = np.sqrt(np.sum(np.square(test_reconstruct - test).reshape(len(test),-1), axis = 1))
sns.distplot(anomality)
pred = anomality
with open(PRED_FOLDER + 'prediction_rawfix_ENC800_'+str(model_type)+'.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(pred)):
        f.write('{},{}\n'.format(i+1, pred[i]))
# %%
# show pics
import matplotlib.pyplot as plt
def show_test_reconstruct(pic):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(test[pic])
    axs[1].imshow(test_reconstruct[pic])

for i in range(20,40,1):
    show_test_reconstruct(i)
#%%
import matplotlib.pyplot as plt
def show_train_reconstruct(pic):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(train[pic])
    axs[1].imshow(train_reconstruct[pic])

for i in range(20,40,1):
    show_train_reconstruct(i)
#%%
# isolation forest

MAX_SAMPLES = 'auto'

print('Isolation Forest fitting ...')
clf = IsolationForest(random_state=SEED, n_jobs=-1)
test_latents = get_latents(test)
clf.fit(test_latents)

print('predicting')
y_score = clf.score_samples(test_latents)
y_pred = clf.predict(test_latents)
print('done')

pred = -y_score * 100
sns.distplot(pred)
# with open(PRED_FOLDER + 'prediction_cnn_iso_'+str(model_type)+'.csv', 'w') as f:
#     f.write('id,anomaly\n')
#     for i in range(len(pred)):
#         f.write('{},{}\n'.format(i+1, pred[i]))

# %%
# by pca
from sklearn.decomposition import PCA, KernelPCA

model_type = 'fcn'
model_specify = 'with_encdec2'

print('PCA fitting')
train_latents = get_latents(train, bottle_neck=False)
# transformer = KernelPCA(n_components=2, kernel='rbf', n_jobs=-1, fit_inverse_transform=True,random_state=SEED)
transformer = PCA(n_components=2, random_state=SEED)
pca = transformer.fit(train_latents)
test_latents = get_latents(test, bottle_neck=False)

y_projected = pca.transform(test_latents)
y_reconstructed = pca.inverse_transform(y_projected)  
dist = np.sqrt(np.sum(np.square(y_reconstructed - test_latents).reshape(len(test_latents), -1), axis=1))

pred = dist
sns.distplot(pred)
# with open(PRED_FOLDER + 'prediction_bottleneck_PCA_'+str(model_type)+'.csv', 'w') as f:
#     f.write('id,anomaly\n')
#     for i in range(len(pred)):
#         f.write('{},{}\n'.format(i+1, pred[i]))

# %%
# testing fit transform
from sklearn.decomposition import PCA, KernelPCA

model_type = 'vae_cnn'
model_specify = 'enc800'

print('PCA fitting')
# transformer = PCA(n_components=2, random_state=SEED)
transformer = KernelPCA(n_components=2, kernel='rbf', n_jobs=-1, fit_inverse_transform=True,random_state=SEED)
test_latents = get_latents(test, bottle_neck=False)

y_projected = transformer.fit_transform(test_latents)
y_reconstructed = transformer.inverse_transform(y_projected)  
dist = np.sqrt(np.sum(np.square(y_reconstructed - test_latents).reshape(len(test_latents), -1), axis=1))

pred = dist
sns.distplot(pred)
# with open(PRED_FOLDER + 'prediction_KPCA_ENC100_'+str(model_type)+'.csv', 'w') as f:
#     f.write('id,anomaly\n')
#     for i in range(len(pred)):
#         f.write('{},{}\n'.format(i+1, pred[i]))

#%%
