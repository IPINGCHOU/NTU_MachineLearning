#%%
import os
# 讀取 label.csv
import pandas as pd
# 讀取圖片
from PIL import Image
import numpy as np

import torch
# Loss function
import torch.nn.functional as F
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as transforms
# 顯示圖片
import matplotlib.pyplot as plt
import random

device = torch.device("cuda")

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#%%
# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

#%%
class Attacker:
    def __init__(self, img_dir, label, models):
        # 讀入預訓練模型 vgg16
        self.model = models
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset('./data/images', label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)


df = pd.read_csv("./data/labels.csv")
df = df.loc[:, 'TrueLabel'].to_numpy()
label_name = pd.read_csv("./data/categories.csv")
label_name = label_name.loc[:, 'CategoryName'].to_numpy()
# new 一個 Attacker class
os.environ['TORCH_HOME'] = './pre_trained_models'
# model = models.vgg19(pretrained = True)
model = models.densenet121(pretrained=True)
attacker = Attacker('./data/images', df, model)


# building one pixel attack
import numpy as np
from scipy.optimize import differential_evolution

def perturb_image(xs, imgs):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = imgs.repeat(batch,1,1,1)

    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x)/5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = r/255.0
            imgs[count, 1, x_pos, y_pos] = g/255.0
            imgs[count, 2, x_pos, y_pos] = b/255.0

        count += 1
    return imgs

def predict_classes(xs, imgs, target_class, model, minimize = True):
    device = 'cuda:0'
    imgs_perturbed = perturb_image(xs, imgs.clone())
    input = imgs_perturbed.to(device)
    predictions = F.softmax(model(input)).data.cpu().numpy()[:, target_class]

    return predictions if minimize else 1-predictions

def attack_success(x, img, target_class, model, targeted_attack = False, verbose = False):
    device = 'cuda:0'
    attack_imgs = perturb_image(x, img.clone())
    input = attack_imgs.to(device)
    confidence = F.softmax(model(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if verbose:
        print("Confidence: %.4f"%confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or (not targeted_attack and predicted_class != target_class):
        return True

def one_pixel_attack(img, label, model, target=None, pixels=1, maxiter=100, popsize=400, verbose = False):
    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

    popmul = max(1, popsize/len(bounds))
    print('popmul : ' + str(popmul))

    predict_fn = lambda xs: predict_classes(
		xs, img, target_class, model, target is None)

    callback_fn = lambda x, convergence: attack_success(
		x, img, target_class, model, targeted_attack, verbose)


    inits = np.zeros([int(popmul*len(bounds)), len(bounds)])

    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*32
            init[i*5+1] = np.random.random()*32
            init[i*5+2] = np.random.normal(128,127)
            init[i*5+3] = np.random.normal(128,127)
            init[i*5+4] = np.random.normal(128,127)
    
    attack_result = differential_evolution(predict_fn,
                                           bounds,
                                           maxiter=maxiter,
                                           popsize=popmul,
                                           recombination=1,
                                           atol=-1,
                                           callback=callback_fn,
                                           polish=False,
                                           init=inits)
    
    device = 'cuda:0'
    attack_imgs = perturb_image(attack_result.x, img)
    attack_var = attack_imgs.to(device)
    predicted_probs = F.softmax(model(attack_var)).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    
    return 0, [None]

#%%
for i, (img, label) in enumerate(attacker.loader):
    print(i)
    flag, x = one_pixel_attack(img, label, model, target=None, pixels=3, maxiter=5, popsize=400, verbose = True)

    if i == 3:
        break 
# too slow, bad performance :(