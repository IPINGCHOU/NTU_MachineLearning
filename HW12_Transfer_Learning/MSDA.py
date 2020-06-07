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

INPUT_TRAIN_DATA = './real_or_drawing/train_data'
INPUT_TEST_DATA = './real_or_drawing/test_data'

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

def euclidean(x1,x2):
	return ((x1-x2)**2).sum().sqrt()


def k_moment(output_s1, output_s2, output_s3, output_s4, output_t, k):
    output_s1 = (output_s1**k).mean(0)
    output_s2 = (output_s2**k).mean(0)
    output_s3 = (output_s3**k).mean(0)
    output_s4 = (output_s4**k).mean(0)
    output_t = (output_t**k).mean(0)
    return euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+ \
		euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
		euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
		euclidean(output_s4, output_t)

def msda_regulizer(output_s1, output_s2, output_s3, output_s4, output_t, belta_moment):
    s1_mean = output_s1.mean(0)
    s2_mean = output_s2.mean(0)
    s3_mean = output_s3.mean(0)
    s4_mean = output_s4.mean(0)
    t_mean = output_t.mean(0)

    output_s1 = output_s1 - s1_mean
    output_s2 = output_s2 - s2_mean
    output_s3 = output_s3 - s3_mean
    output_s4 = output_s4 - s4_mean
    output_t = output_t - t_mean

    moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+\
        euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) +\
        euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
        euclidean(output_s4, output_t)
    
    reg_info = moment1
    for i in range(belta_moment-1):
        reg_info += k_moment(output_s1,output_s2,output_s3, output_s4, output_t,i+2)

    return reg_info/6

#%%
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

Canny_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Canny_transform)
Sobel_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Sobel_transform)
Laplacian_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Laplacian_transform)
Gray_dataset = ImageFolder(INPUT_TRAIN_DATA, transform=Gray_transform)
target_dataset = ImageFolder(INPUT_TEST_DATA, transform=target_transform)

Canny_dataloader = DataLoader(Canny_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
Sobel_dataloader = DataLoader(Sobel_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
Laplacian_dataloader = DataLoader(Laplacian_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
Gray_dataloader = DataLoader(Gray_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
target_dataloader = DataLoader(target_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

#%%
# models
# as tutorial
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

# new
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

G = FeatureExtractor().cuda()
C1 = Predictor().cuda()
C2 = Predictor().cuda()

criterion = nn.CrossEntropyLoss().cuda()

optim_g  = optim.Adam(G.parameters(), lr = 0.001)
optim_c1 = optim.Adam(C1.parameters(), lr = 0.001)
optim_c2 = optim.Adam(C2.parameters(), lr = 0.001)

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def feat_all_domain(img_s1, img_s2, img_s3, img_s4, img_t):
    return G(img_s1), G(img_s2), G(img_s3), G(img_s4), G(img_t)

def C1_all_domain(feat1, feat2,feat3,feat4, feat_t):
    return C1(feat1), C1(feat2), C1(feat3), C1(feat4), C1(feat_t)

def C2_all_domain(feat1, feat2,feat3,feat4, feat_t):
    return C2(feat1), C2(feat2), C2(feat3), C2(feat4), C2(feat_t) 

def softmax_loss_all_domain(output1, output2, output3, output4, label_s1, label_s2, label_s3, label_s4):
    criterion = nn.CrossEntropyLoss()
    return criterion(output1, label_s1), criterion(output2, label_s2), criterion(output3, label_s3), criterion(output4,label_s4)

def loss_all_domain(img_s1, img_s2, img_s3, img_s4, img_t, label_s1, label_s2, label_s3, label_s4):
    feat_s1, feat_s2, feat_s3, feat_s4, feat_t = feat_all_domain(img_s1, img_s2, img_s3, img_s4, img_t)
    output_s1_c1, output_s2_c1, output_s3_c1, output_s4_c1, output_t_c1 = \
        C1_all_domain(feat_s1, feat_s2, feat_s3, feat_s4, feat_t)
    output_s1_c2, output_s2_c2, output_s3_c2, output_s4_c2, output_t_c2 = \
        C2_all_domain(feat_s1,feat_s2, feat_s3, feat_s4, feat_t)
    loss_msda =  0.0005 * msda_regulizer(feat_s1, feat_s2, feat_s3, feat_s4, feat_t, 5)
    loss_s1_c1, loss_s2_c1,loss_s3_c1,loss_s4_c1 =\
        softmax_loss_all_domain(output_s1_c1, output_s2_c1, output_s3_c1,output_s4_c1, label_s1, label_s2, label_s3,label_s4)
    loss_s1_c2, loss_s2_c2,loss_s3_c2,loss_s4_c2 =\
        softmax_loss_all_domain(output_s1_c2, output_s2_c2, output_s3_c2,output_s4_c2, label_s1, label_s2, label_s3,label_s4)
    return  loss_s1_c1, loss_s2_c1,loss_s3_c1,loss_s4_c1, loss_s1_c2, loss_s2_c2,loss_s3_c2, loss_s4_c2, loss_msda


def reset_grad():
    optim_g.zero_grad()
    optim_c1.zero_grad()
    optim_c2.zero_grad()

def train_MSDA():
    G.train()
    C1.train()
    C2.train()
    mean_loss_c1, mean_loss_c2, mean_loss_dis = 0.0, 0.0, 0.0
    s1_acc, s2_acc, s3_acc, s4_acc, total_num = 0.0, 0.0, 0.0, 0.0, 0.0

    for i, ((Canny_data, Canny_label), (Sobel_data, Sobel_label),
        (Laplacian_data, Laplacian_label), (Gray_data, Gray_label), (target_data, _)) in\
        enumerate(zip(Canny_dataloader, Sobel_dataloader, Laplacian_dataloader, Gray_dataloader, target_dataloader)):
        img_t = target_data.cuda()
        img_s1 = Canny_data.cuda()
        img_s2 = Sobel_data.cuda()
        img_s3 = Laplacian_data.cuda()
        img_s4 = Gray_data.cuda()
        label_s1 = Canny_label.long().cuda()
        label_s2 = Sobel_label.long().cuda()
        label_s3 = Laplacian_label.long().cuda()
        label_s4 = Gray_label.long().cuda()

        reset_grad()

        loss_s1_c1, loss_s2_c1, loss_s3_c1, loss_s4_c1, loss_s1_c2, loss_s2_c2, loss_s3_c2, loss_s4_c2, loss_msda = \
            loss_all_domain(img_s1, img_s2, img_s3, img_s4, img_t, label_s1, label_s2, label_s3, label_s4)

        loss_s_c1 = loss_s1_c1 + loss_s2_c1 + loss_s3_c1 + loss_s4_c1
        loss_s_c2 = loss_s1_c2 + loss_s2_c2 + loss_s3_c2 + loss_s4_c2
        loss = loss_s_c1 + loss_s_c2 + loss_msda

        loss.backward()
        
        optim_g.step()
        optim_c1.step()
        optim_c2.step()

        reset_grad()

        loss_s1_c1, loss_s2_c1, loss_s3_c1, loss_s4_c1, loss_s1_c2, loss_s2_c2, loss_s3_c2, loss_s4_c2, loss_msda = loss_all_domain(img_s1, img_s2, img_s3, img_s4, img_t, label_s1, label_s2, label_s3,label_s4)

        feat_t = G(img_t)
        output_t1 = C1(feat_t)
        output_t2 = C2(feat_t)
        loss_s_c1 = loss_s1_c1 + loss_s2_c1 + loss_s3_c1 + loss_s4_c1
        loss_s_c2 = loss_s1_c2 + loss_s2_c2 + loss_s3_c2 + loss_s4_c2

        loss_s = loss_s1_c1 + loss_s2_c2 + loss_msda
        loss_dis = discrepancy(output_t1, output_t2)
        loss = loss_s - loss_dis
        loss.backward()
        optim_c1.step()
        optim_c2.step()
        reset_grad()

        for i in range(4):
            feat_t = G(img_t)
            output_t1 = C1(feat_t)
            output_t2 = C2(feat_t)
            loss_dis = discrepancy(output_t1, output_t2)
            loss_dis.backward()
            optim_g.step()
            reset_grad()

        with torch.no_grad():
            feat_s1 = G(img_s1)
            feat_s2 = G(img_s2)
            feat_s3 = G(img_s3)
            feat_s4 = G(img_s4)
            out_s1_c1 = C1(feat_s1)
            out_s2_c1 = C1(feat_s2)
            out_s3_c1 = C1(feat_s3)
            out_s4_c1 = C1(feat_s4)

            s1_acc += torch.sum(torch.argmax(out_s1_c1, dim=1) == label_s1).item()
            s2_acc += torch.sum(torch.argmax(out_s2_c1, dim=1) == label_s2).item()
            s3_acc += torch.sum(torch.argmax(out_s3_c1, dim=1) == label_s3).item()
            s4_acc += torch.sum(torch.argmax(out_s4_c1, dim=1) == label_s4).item()

            total_num += Canny_data.shape[0]
            mean_loss_c1 += loss_s_c1.item()
            mean_loss_c2 += loss_s_c2.item() 
            mean_loss_dis += loss_dis.item()
        
    return mean_loss_c1 / (i+1), mean_loss_c2 / (i+1), mean_loss_dis / (i+1), \
           s1_acc / total_num, s2_acc / total_num, s3_acc / total_num, s4_acc / total_num 

#%%
best_loss = np.inf
MAX_EPOCH = 500


for epoch in range(MAX_EPOCH):
    c1_loss, c2_loss, g_loss, s1_acc, s2_acc, s3_acc, s4_acc  = train_MSDA()
    
    print('epoch {:>3d}: c1 loss: {:6.4f}, c2 loss: {:6.4f}, g loss: {:6.4f}, s1_acc {:6.4f}, s2_acc {:6.4f}, s3_acc {:6.4f}, s4_acc {:6.4f}'.format(epoch,
    c1_loss, c2_loss, g_loss, s1_acc, s2_acc, s3_acc, s4_acc))

    loss = c1_loss + c2_loss + g_loss 
    if loss < best_loss:
        best_loss = loss 
        torch.save(G.state_dict(), f'G.bin')
        torch.save(C1.state_dict(), f'C1.bin')
        torch.save(C2.state_dict(), f'C2.bin')
        print("Model Saved!!!")


    # print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))
#%%