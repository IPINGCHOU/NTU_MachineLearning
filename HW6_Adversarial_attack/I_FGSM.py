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
            if pic_id % 50 == 0:
                print('Now processing: ' + str(pic_id))

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

#%%
import os
import time
if __name__ == '__main__':
    # 讀入圖片相對應的 label
    df = pd.read_csv("./data/labels.csv")
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv("./data/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    os.environ['TORCH_HOME'] = './pre_trained_models'
    # model = models.vgg19(pretrained = True)
    model = models.densenet121(pretrained=True)
    attacker = Attacker('./data/images', df, model)
    # 要嘗試的 epsilon
    # epsilons = [0.4,0.3,0.2,0.1]
    # epsilons = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

    epsilons = 0.03
    alphas = [0.005,0.004,0.003,0.002,0.001]
    iteration = 15
    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    # for eps in epsilons:
    #     ex, acc, _ = attacker.attack(alpha,eps,iteration)
    #     accuracies.append(acc)
    #     examples.append(ex)

    for alpha in alphas:
        print('alpha: ' + str(alpha))
        eps = epsilons
        start_time = time.time()
        ex, acc, _ = attacker.attack(alpha,eps,iteration)
        print('Elasped: ' + str(time.time() - start_time))
        accuracies.append(acc)
        examples.append(ex)

#%%
#%%
cnt = 0
epsilons = alphas
plt.figure(figsize=(30, 30))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,orig_img, ex = examples[i][j]
        # plt.title("{} -> {}".format(orig, adv))
        plt.title("original: {}".format(label_name[orig].split(',')[0]))
        orig_img = np.transpose(orig_img, (1, 2, 0))
        plt.imshow(orig_img)
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
        plt.title("adversarial: {}".format(label_name[adv].split(',')[0]))
        ex = np.transpose(ex, (1, 2, 0))
        plt.imshow(ex)
plt.tight_layout()
plt.show()

#%%
# generate attacked images
model = models.densenet121(pretrained=True)
attacker = Attacker('./data/images', df, model)


eps = 0.03
alpha = 0.005
iteration = 5
ex, acc, all_attacked = attacker.attack(eps,alpha, iteration, save_all=True)
#%%
# saving data
folder_name = 'attack_densenet121_IFGSM_eps0.03_a0.002_iter15'
from os import path
import matplotlib.pyplot as plt

if path.exists(folder_name) == False:
    os.mkdir(folder_name)

for i, pic in enumerate(all_attacked):
    pic = np.moveaxis(pic, 0, -1)
    plt.imsave(os.path.join(folder_name, "{0:0=3d}.png".format(i)), np.clip(pic,0,1))

# densenet 121 , eps = 0.3, match acc on judgeboi, with 0.93 acc and 16.7250 L-inf


# %%

# first eps 0.03 alpha 0.005 iter 5 , acc = 1 , l-inf = 8.4
# second eps 0.03 alpha 0.002 iter 15, acc = 1, l-inf = 24.995

#%%
# report Q3
# get predict prob before attack
# get 3 pictures from loader
# label_name : all label name
images = []
labels = []

for pic_id, (data, target) in enumerate(attacker.loader):
    images.append(data)
    labels.append(target)
# ============
indices = [0,100,180]
# ============
images = [images[i] for i in indices]
labels = [labels[i] for i in indices]

# %%
# get model output before attacke and after
output_list_before_attack = []
output_list_after_attack = []
for pic_id, (image,label) in enumerate(zip(images, labels)):
    image = image.to(device)
    label = label.to(device)

    output =  model(image)
    output_list_before_attack.append(F.softmax(output)[0].data.cpu().numpy())

    perturbed_pic = attacker.I_fgsm_attack(image, label, eps=0.03, alpha=0.005, iteration=5 )
    perturbed_output = model(perturbed_pic)
    output_list_after_attack.append(F.softmax(perturbed_output)[0].data.cpu().numpy())

#%%
# get top3 prob and name
before_top3_prob = []
after_top3_prob = []
before_top3_label = []
after_top3_label = []
for i in range(3):
    before_top3_pos = output_list_before_attack[i].argsort()[-3:][::-1]
    after_top3_pos = output_list_after_attack[i].argsort()[-3:][::-1]
    
    before_top3_prob.append(output_list_before_attack[i][before_top3_pos])
    after_top3_prob.append(output_list_after_attack[i][after_top3_pos])

    before_top3_label.append([i.split(',')[0] for i in label_name[before_top3_pos]])
    after_top3_label.append([i.split(',')[0] for i in label_name[after_top3_pos]])
    
# %%
fig, axs = plt.subplots(3,3,figsize = (30,30))

for row in range(3):
    # fix image var and mean
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = images[row][0]
    image = image *  torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    image = image.permute(1,2,0).data.cpu().numpy()
    
    axs[row][0].imshow(image)
    axs[row][1].bar(before_top3_label[row], before_top3_prob[row])
    axs[row][1].tick_params(axis="x", labelsize=25)
    axs[row][1].tick_params(axis="y", labelsize=25)
    axs[row][1].set_xticklabels(before_top3_label[row],rotation=10)
    axs[row][2].bar(after_top3_label[row], after_top3_prob[row])
    axs[row][2].tick_params(axis="x", labelsize=25)
    axs[row][2].tick_params(axis="y", labelsize=25)
    axs[row][2].set_xticklabels(after_top3_label[row],rotation=10)

plt.show()

# %%
# report Q4
# gaussian filiter
# check attacked imgs first!
# all_attacked : attacked pic
# =========
indices = [0,1,2]
# =========
fig, axs = plt.subplots(1,3, figsize = (30,30))
for i, ax in enumerate(axs):
    ax.imshow(np.transpose(all_attacked[indices[i]], (1,2,0)))
#%%
# prepare for gaussian kernel
import math
x, y = np.mgrid[-1:2, -1:2]
sigma = 5
gaussian_kernel = np.exp(-(x**2+y**2)/(2*sigma**2)) / (2*sigma**2*math.pi)
#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()
print(gaussian_kernel)

# filtering
from scipy import signal
from scipy import misc
filtered_images = []

for pic_id, image in enumerate(all_attacked):
    image = np.transpose(image, (1,2,0))
    r = signal.convolve2d(image[:,:,0], gaussian_kernel, mode='same')
    g = signal.convolve2d(image[:,:,1], gaussian_kernel, mode='same')
    b = signal.convolve2d(image[:,:,2], gaussian_kernel, mode='same')
    filtered_image = np.dstack([r,g,b])
    
    filtered_images.append(filtered_image)
print('g-filtered done')
# %%
# plot the filtered_images
fig, axs = plt.subplots(1,3, figsize = (30,30))
for i, ax in enumerate(axs):
    ax.imshow(filtered_images[indices[i]])

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# df : label
normalize = transforms.Normalize(mean, std, inplace=False)
transform = transforms.Compose([                
                transforms.ToTensor(),
                normalize
            ])

match = 0
for pic_id, (f_image, label) in enumerate(zip(filtered_images, df)):
    f_image = transform(f_image)
    f_image = f_image.to(device).unsqueeze(0).float()

    output = model(f_image)
    final_pred = output.max(1, keepdim=True)[1]
    
    if final_pred.item() == label:
        match+=1

print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(0.005, match, 200, match/200))

    
# %%
