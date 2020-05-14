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
MODEL_SAVEFOLDER = sys.argv[2]

#%%
# 主要的訓練過程
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_SAVEFOLDER + '/checkpoint_{}.pth'.format(epoch+1))
            
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

# 訓練完成後儲存 model
torch.save(model.state_dict(), MODEL_SAVEFOLDER+'/improved_2.pth')

#%%
