import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import random

random.seed(1024)
np.random.seed(1024)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            # VGG 16
            nn.Conv2d(3, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 64 64

            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # 32 * 32

            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # 16 * 16

            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     # 8 * 8
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),    # 4 * 4 
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


import sys
args = {
      'ckptpath': './CNN_model_best.pkl',
      'dataset_dir': sys.argv[1]
}
args = argparse.Namespace(**args)
output_path = sys.argv[2]
#%%

# model = Classifier().cuda()
model = torch.load(args.ckptpath)
model = model.cuda()
#%%
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        else:
            X = torch.Tensor(X)
            width,height,channel = X.shape
            X = X.view(channel,width,height)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

    def getbatch(self, indices):
        images, labels = [], []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

import sys
workspace_dir = sys.argv[1]
print("Reading data")
from torch.utils.data import DataLoader, Dataset
# train_paths, train_labels = get_paths_labels(os.path.join(args.dataset_dir, 'training'))
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
train_set = ImgDataset(train_x, train_y, test_transform)
# val_set = ImgDataset(val_x,val_y, None)
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#%%
# get shap plots
# soup category
import shap
# gather test images
# =========================
# report Q4 label 9
img_indices = [9008,9009,9010,9011]
# =========================

images, labels = train_set.getbatch(img_indices)
# gather train background images
back_images, _ = next(iter(train_loader))

explainer = shap.DeepExplainer(model, back_images.cuda())
shap_values = explainer.shap_values(images)

for i,img in enumerate(images):
    convert_img = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    images[i] = torch.FloatTensor(convert_img).permute(2,0,1)

# shape in shap list is (4,3,128,128) -> (4,3,128,128)
for i, sv in enumerate(shap_values):
    sv = np.swapaxes(sv, 2, 1)
    sv = np.swapaxes(sv, 3, 2)
    shap_values[i] = sv
# %%
shap.image_plot(shap_values, images.data.permute(0,2,3,1).numpy(), show = False)

print('saving fig 6')
plt.savefig(os.path.join(output_path, '6.png'))

# =========================
# report Q4 label 2
img_indices = [2001,2002,2003,2004]
# =========================
images, labels = train_set.getbatch(img_indices)
# gather train background images
back_images, _ = next(iter(train_loader))

explainer = shap.DeepExplainer(model, back_images.cuda())
shap_values = explainer.shap_values(images)

for i,img in enumerate(images):
    convert_img = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    images[i] = torch.FloatTensor(convert_img).permute(2,0,1)

# shape in shap list is (4,3,128,128) -> (4,3,128,128)
for i, sv in enumerate(shap_values):
    sv = np.swapaxes(sv, 2, 1)
    sv = np.swapaxes(sv, 3, 2)
    shap_values[i] = sv
# %%
shap.image_plot(shap_values, images.data.permute(0,2,3,1).numpy(), show = False)

print('saving fig 7')
plt.savefig(os.path.join(output_path, '7.png'))

print('done')