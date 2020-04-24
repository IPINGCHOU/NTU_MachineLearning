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
from torch.autograd import Variable

device = torch.device("cuda")

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

import sys
pic_folder = sys.argv[1]
output_folder = sys.argv[2]

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
        self.check_grad = 0
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

    def where(self, cond, x, y):
        cond = cond.float()
        return (cond*x) + ((1-cond)*y)

    # FGSM 攻擊
    def I_fgsm_attack(self, data, target, eps=0.03, alpha=1, iteration=10, x_val_min=0, x_val_max=1):
        data = Variable(data.data, requires_grad = True)

        for i in range(iteration):
            output = self.model(data)
            cost = -F.nll_loss(output, target)

            self.model.zero_grad()
            if data.grad is not None:
                data.grad.data.fill_(0)
            cost.backward()

            self.check_grad = data.grad
            data.grad.sign_()

            data = data - alpha*data.grad
            data = self.where(data > data+eps, data+eps, data)
            data = self.where(data < data-eps, data-eps, data)
            # data = torch.clamp(data, x_val_min, x_val_max)
            data = Variable(data.data, requires_grad =True)

        return data
        
    
    def attack(self, alpha, epsilon, iteration, save_all = False):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        all_attacked = []
        wrong, fail, success = 0, 0, 0
        for pic_id, (data, target) in enumerate(self.loader):
            print('Now processing: ' + str(pic_id), end = '\r')

            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                all_attacked.append(data_raw)
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行 FGSM 攻擊
            perturbed_data = self.I_fgsm_attack(data, target, eps=epsilon, alpha=alpha, iteration=iteration)

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            if save_all == True:
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                all_attacked.append(adv_ex)
          
            if final_pred.item() == target.item():
                # 辨識結果還是正確 攻擊失敗
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                # 將攻擊成功的圖片存入
                if len(adv_examples) < 5:
                  adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
                  data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                  data_raw = data_raw.squeeze().detach().cpu().numpy()
                  adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )        
        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, final_acc, all_attacked


df = pd.read_csv(os.path.join(pic_folder,'labels.csv'))
df = df.loc[:, 'TrueLabel'].to_numpy()
label_name = pd.read_csv(os.path.join(pic_folder,'categories.csv'))
label_name = label_name.loc[:, 'CategoryName'].to_numpy()
model = models.densenet121(pretrained=True)
attacker = Attacker(os.path.join(pic_folder, 'images'), df, model)


eps = 0.03
alpha = 0.005
iteration = 5
print('eps: ' + str(eps))
print('alpha: ' + str(alpha))
print('iteraion: ' + str(iteration))
print('attacking')
ex, acc, all_attacked = attacker.attack(eps,alpha, iteration, save_all=True)


from os import path
import matplotlib.pyplot as plt

print('saving')
for i, pic in enumerate(all_attacked):
    pic = np.moveaxis(pic, 0, -1)
    plt.imsave(os.path.join(output_folder, "{0:0=3d}.png".format(i)), np.clip(pic,0,1))