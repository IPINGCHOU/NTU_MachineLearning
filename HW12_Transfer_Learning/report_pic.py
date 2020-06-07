import numpy as np
import cv2

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

SEED = 1234
same_seeds(1234)

#%%
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

# new
# 
class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Dropout(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

#%%
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

INPUT_TEST_DATA = './real_or_drawing/test_data'

C1_C2 = 'C1'

# data transforms and dataloader
# Canny transform
Canny_transform = transforms.Compose([
    transforms.Grayscale(),
    # cv2.Canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# Sobel
Sobel_transform = transforms.Compose([
    transforms.Grayscale(),
    # cv2.Sobel
    transforms.Lambda(lambda x: cv2.addWeighted(\
        cv2.convertScaleAbs(cv2.Sobel(np.array(x),cv2.CV_16S,1,0, ksize=3)),0.5, \
        cv2.convertScaleAbs(cv2.Sobel(np.array(x),cv2.CV_16S,0,1, ksize=3)),0.5,0)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# Laplacian
Laplacian_transform = transforms.Compose([
    transforms.Grayscale(),
    # cv2.Laplacian
    transforms.Lambda(lambda x: cv2.convertScaleAbs(cv2.Laplacian(np.array(x), cv2.CV_16S, ksize=3))),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# Gray only
Gray_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# Target
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

INPUT_TRAIN_DATA = './real_or_drawing/train_data'
INPUT_TEST_DATA = './real_or_drawing/test_data'

Canny_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Canny_transform)
Sobel_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Sobel_transform)
Laplacian_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Laplacian_transform)
Gray_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Gray_transform)
target_dataset = ImageFolder(INPUT_TEST_DATA, transform=target_transform)

Canny_dataloader = DataLoader(Canny_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
Sobel_dataloader = DataLoader(Sobel_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
Laplacian_dataloader = DataLoader(Laplacian_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
Gray_dataloader = DataLoader(Gray_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, drop_last=False)
target_dataloader = DataLoader(target_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)


G = FeatureExtractor().cuda()
C = Predictor().cuda()

G.load_state_dict(torch.load("G.bin"))


if C1_C2 == 'C1':
    C.load_state_dict(torch.load("C1.bin"))
else:
    C.load_state_dict(torch.load("C2.bin"))
#%%

canny_features, sobel_features, laplacian_features, gray_features = [], [], [], []
canny_raw, sobel_raw, laplacian_raw, gray_raw = [], [], [], []
G.eval()

for i, ((Canny_data, Canny_label), (Sobel_data, Sobel_label),
    (Laplacian_data, Laplacian_label), (Gray_data, Gray_label)) in\
    enumerate(zip(Canny_dataloader, Sobel_dataloader, Laplacian_dataloader, Gray_dataloader)):
    img_canny = Canny_data.cuda()
    img_sobel = Sobel_data.cuda()
    img_laplacian = Laplacian_data.cuda()
    img_gray = Gray_data.cuda()
    canny_raw.append(img_canny.detach().cpu().numpy())
    sobel_raw.append(img_sobel.detach().cpu().numpy())
    laplacian_raw.append(img_laplacian.detach().cpu().numpy())
    gray_raw.append(img_gray.detach().cpu().numpy())

    canny_f = G(img_canny)
    sobel_f = G(img_sobel)
    laplacian_f = G(img_laplacian)
    gray_f = G(img_gray)

    canny_features.append(canny_f.detach().cpu().numpy())
    sobel_features.append(sobel_f.detach().cpu().numpy())
    laplacian_features.append(laplacian_f.detach().cpu().numpy())
    gray_features.append(gray_f.detach().cpu().numpy())


canny_features = np.array(canny_features)
canny_features = np.concatenate(canny_features, axis = 0)
sobel_features = np.array(sobel_features)
sobel_features = np.concatenate(sobel_features, axis = 0)
laplacian_features = np.array(laplacian_features)
laplacian_features = np.concatenate(laplacian_features, axis = 0)
gray_features = np.array(gray_features)
gray_features = np.concatenate(gray_features, axis = 0)

canny_raw = np.concatenate(canny_raw, axis = 0)
canny_raw = canny_raw.reshape(len(canny_raw), -1)
sobel_raw = np.concatenate(sobel_raw, axis = 0)
sobel_raw = sobel_raw.reshape(len(sobel_raw), -1)
laplacian_raw = np.concatenate(laplacian_raw, axis = 0)
laplacian_raw = laplacian_raw.reshape(len(laplacian_raw), -1)
gray_raw = np.concatenate(gray_raw, axis = 0)
gray_raw = gray_raw.reshape(len(gray_raw), -1)

test_features = []
test_raw = []
for i, (test_data, _) in enumerate(target_dataloader):
    test_data = test_data.cuda()
    test_raw.append(test_data.detach().cpu().numpy())

    # get features
    features = G(test_data)

    test_features.append(features.detach().cpu().numpy())

test_features = np.array(test_features)
test_features = np.concatenate(test_features, axis = 0)
test_raw = np.concatenate(test_raw, axis = 0)
test_raw = test_raw.reshape(len(test_raw), -1)

# %%
# PCA for dimension reduction
# tsne for plotting
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca_features = PCA(n_components = 2).fit(np.concatenate((canny_features, sobel_features, laplacian_features, gray_features, test_features), axis = 0))
print('Feature pca_features fitting...')
canny_f_pca_features_transform = pca_features.transform(canny_features)
sobel_f_pca_features_transform = pca_features.transform(sobel_features)
laplacian_f_pca_features_transform = pca_features.transform(laplacian_features)
gray_f_pca_features_transform = pca_features.transform(gray_features)
test_f_pca_transform = pca_features.transform(test_features)

print('Raw pics pca_features fitting...')
pca_raw = PCA(n_components = 2).fit(np.concatenate((canny_raw, sobel_raw, laplacian_raw, gray_raw, test_raw), axis = 0))
canny_raw_pca_transform = pca_raw.transform(canny_raw)
sobel_raw_pca_transform = pca_raw.transform(sobel_raw)
laplacian_raw_pca_transform = pca_raw.transform(laplacian_raw)
gray_raw_pca_transform = pca_raw.transform(gray_raw)
test_raw_pca_transform = pca_raw.transform(test_raw)
# %%
# before domain adversarial training
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize = (20,10))

axs.scatter(test_raw_pca_transform[:,0], test_raw_pca_transform[:,1], s = 10, label = 'target')
axs.scatter(canny_raw_pca_transform[:,0], canny_raw_pca_transform[:,1], s = 10, label = 'canny')
axs.scatter(sobel_raw_pca_transform[:,0], sobel_raw_pca_transform[:,1], s = 10, label = 'sobel')
axs.scatter(laplacian_raw_pca_transform[:,0], laplacian_f_pca_transform[:,1], s = 10, label = 'laplacian')
axs.scatter(gray_raw_pca_transform[:,0], gray_raw_pca_transform[:,1], s = 10, label = 'gray')
axs.legend(fontsize=15)
axs.set_title('Before domain adversarial training', fontsize = 30)
fig.savefig('before_dat.png')


# %%
fig, axs = plt.subplots(figsize = (20,10))

axs.scatter(test_f_pca_transform[:,0], test_f_pca_transform[:,1], s = 10, label = 'target')
axs.scatter(canny_f_pca_transform[:,0], canny_f_pca_transform[:,1], s = 10, label = 'canny')
axs.scatter(sobel_f_pca_transform[:,0], sobel_f_pca_transform[:,1], s = 10, label = 'sobel')
axs.scatter(laplacian_f_pca_transform[:,0], laplacian_f_pca_transform[:,1], s = 10, label = 'laplacian')
axs.scatter(gray_f_pca_transform[:,0], gray_f_pca_transform[:,1], s = 10, label = 'gray')
axs.legend(fontsize = 15)
axs.set_title('After domain adversarial training', fontsize = 30)
fig.savefig('after_dat.png')

# %%
