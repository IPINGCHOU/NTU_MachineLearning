import numpy as np

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images
#%%
from torch.utils.data import DataLoader
import os
import sys


trainX_path = sys.argv[1]

trainX = np.load(trainX_path)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

#%%
import random
import torch

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#%%
# baseline autoencoder
# same architecture as Hinton
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2), # 1024 * 2 * 2
        )

 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 7, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 10, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 12, stride=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(128, 64, 8, stride=1),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 3, 7, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x

#%%
# training 
import torch
from torch import optim

same_seeds(0)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 200

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=1024, shuffle=True)

# === h para ===
MODEL_SAVEFOLDER = './improve2_model'

#%%
import numpy as np

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)

import matplotlib.pyplot as plt

def plot_scatter(feat, label, dot_size = 1,savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0] 
    Y = feat[:, 1] 
    plt.scatter(X, Y, c = label, s = dot_size)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, KMeans

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents
#%%
def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=KPCA_comp, kernel='rbf', n_jobs=-1, random_state=SK_SEED)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter = TSNE_ITER, init = 'pca', learning_rate=TSNE_LR, random_state=SK_SEED).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    # KMeans mini
    # pred = MiniBatchKMeans(n_clusters=2, random_state=SK_SEED).fit(X_embedded)
    # KMeans
    pred = KMeans(n_clusters=2, random_state=SK_SEED).fit(X_embedded)
    
    # Spectral Clustering
    # pred = SpectralClustering(n_clusters=2, random_state=SK_SEED, affinity=SP_AFFINITY).fit(X_embedded)

    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

#%%
# =====================
SK_SEED = 0
TSNE_PERPLEXITY = 100
TSNE_ITER = 1000
TSNE_LR = 200
# SP_AFFINITY = 'nearest_neighbors'
KPCA_comp = 200

#%%
# load model

model = AE().cuda()
model.load_state_dict(torch.load(sys.argv[2]+'/improved_2.pth'))
model.eval()

# 準備 data
trainX = np.load(trainX_path)

# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

#%%
# 將預測結果存檔，上傳 kaggle
save_prediction(pred, sys.argv[3])
print('done')



