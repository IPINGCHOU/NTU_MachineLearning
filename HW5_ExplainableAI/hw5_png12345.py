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


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # 最關鍵的一行 code
    # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
    # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
    # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
    # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗，
    # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
    # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

# =========================
# report Q1
img_indices = [454,87,1543,3350]
# =========================
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

for i,img in enumerate(images):
    convert_img = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    images[i] = torch.FloatTensor(convert_img).permute(2,0,1)

# 使用 matplotlib 畫出來
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img.permute(1, 2, 0).numpy())

import os
output_path = sys.argv[2]
print('saving fig 1')
plt.savefig(os.path.join(output_path, '1.png'))

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
    # cnnid, filterid: 想要指定第幾層 cnn 中第幾個 filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output
  
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義的 function 後才可以繼續 forward 下一層 cnn
    # 因此上面的 hook function 中，我們就會把該層的 output，也就是 activation map 記錄下來，這樣 forward 完整個 model 後我們就不只有 loss
    # 也有某層 cnn 的 activation map
    # 注意：到這行為止，都還沒有發生任何 forward。我們只是先告訴 pytorch 等下真的要 forward 時該多做什麼事
    # 注意：hook_handle 可以先跳過不用懂，等下看到後面就有說明了

    # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
    model(x.cuda())
    # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
    # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor

    # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
    x = x.cuda()
    # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
    x.requires_grad_()
    # 我們要對 input image 算偏微分
    optimizer = Adam([x], lr=lr)
    # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)

        objective = -layer_activations[:, filterid, :, :].sum()
        # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
        # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
        # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization

        objective.backward(retain_graph=True)
        # 計算 filter activation 對 input image 的偏微分
        optimizer.step()
        # 修改 input image 來最大化 filter activation
        filter_visualization = x.detach().cpu().squeeze()[0]
        # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

        hook_handle.remove()
        # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
        # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
        # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

    return filter_activations, filter_visualization


# =========================
# report Q2
test_filter = [0,15,30,45,63]
img_indices = [454,87,1543,3350]
# =========================

images, labels = train_set.getbatch(img_indices)
fig,axs = plt.subplots(1,len(test_filter), figsize = (15,10))
for i, filter_id in enumerate(test_filter):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=3, filterid=filter_id, iteration=100, lr=0.1)
    axs[i].imshow(filter_visualization.permute(1,2,0))

print('saving fig 2')
plt.savefig(os.path.join(output_path, '2.png'))

images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(len(test_filter)+1, len(img_indices), figsize=(10, 10))
for i, filter_id in enumerate(test_filter):
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=3, filterid=filter_id, iteration=100, lr=0.1)
    for j, img in enumerate(filter_activations):
        axs[i+1][j].imshow(normalize(img))

for i,img in enumerate(images):
    convert_img = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    images[i] = torch.FloatTensor(convert_img).permute(2,0,1)
for i, img in enumerate(images):
    axs[0][i].imshow(img.permute(1, 2, 0))

print('saving fig 3')
plt.savefig(os.path.join(output_path, '3.png'))

def predict(input):
    # input: numpy array, (batches, height, width, channels)                                                                                                                                                     
    
    model.eval()                                                                                                                                                             
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                            
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.cuda())                                                                                                                             
    return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                             
def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊                                                                                                                                      
    return slic(input, n_segments=100, compactness=1, sigma=1)                                                                                                              
                                                                                                                                                                             

# =========================
# report Q3 label 9
img_indices = [9008,9009,9010,9011]
# =========================

def get_lime(img_indices, filename):
    images, labels = train_set.getbatch(img_indices)
    fig,axs = plt.subplots(2,4, figsize = (15,8))
    for i, img in enumerate(images):
        convert_img = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
        axs[0][i].imshow(convert_img)
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image = x, classifier_fn=predict, segmentation_fn = segmentation)

        lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                label=label.item(),                                                                                                                           
                                positive_only=False,                                                                                                                         
                                hide_rest=False,                                                                                                                             
                                num_features=11,                                                                                                                              
                                min_weight=0.05                                                                                                                              
                            )
        
        axs[1][idx].imshow(lime_img)
    
    plt.savefig(os.path.join(output_path, filename))
    print(labels)

print('saving fig 4')
get_lime(img_indices, '4.png')

# =========================
# report Q3 label 1
img_indices = [2001,2002,2003,2004]
# =========================
print('saving fig 5')
get_lime(img_indices, '5.png')

print('png 1 2 3 4 5 done')
print('Reload data and model for releasing GPU memory')
