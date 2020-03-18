#%%
import sys
import pandas as pd
import numpy as np

# the data reading should be transfer to shell for .sh file 
data = pd.read_csv('train.csv', encoding = 'big5')

#%%
# Prepocessing
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy() # 4320 * 18 (features = 18)

# transfer 4320*18 to 12 month (480(hours) * 18(feature) * 12(month))
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

#%%
# set a window with size = 10 datas, 1~9th for x and 10th for y
window_size = 9
def xy_Window(window_size = 9):
    left = 480-window_size
    x = np.empty([12 * left, 18 * window_size], dtype = float)
    y = np.empty([12 * left, 1], dtype = float)    

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * left + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + window_size].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * left + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + window_size] #value
    return x,y

x,y = xy_Window(window_size)

#%%
# Normalizing
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


#%%
# Spliting training and valid set
import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

#%%
# Problem 1
# Training with Adagrad
# Testing different learning rate
dim = 18 * 9 + 1

x_train_set_length = np.shape(x_train_set)[0]
x_train = np.concatenate((np.ones([x_train_set_length, 1]), x_train_set), axis = 1).astype(float)
learning_rate_list = [1,10,50,100,200,500]
iter_time = 100000
eps = 1e-7

adagrad_loss_list = []
for lr in learning_rate_list:
    #initialize
    w = np.zeros([dim, 1])
    adagrad = np.zeros([dim, 1])
    cur_loss = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/x_train_set_length)#rmse
        cur_loss.append(loss)
        if(t%5000==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set) #dim*1
        adagrad += gradient ** 2
        w = w - lr * gradient / np.sqrt(adagrad + eps)
    adagrad_loss_list.append(cur_loss)
    print(str(t) + ":" + str(loss))
# all loss by adagrad were stored in adagrad_loss_list
# tested lr : [1,10,50,100,200,500]

#%%
# Problem 2
# take only 5 datas instead of 9 datas in an iteration
# a moving window function for cuting datas
window_size = 5
def xy_Window(window_size = 9):
    left = 480-window_size
    x = np.empty([12 * left, 18 * window_size], dtype = float)
    y = np.empty([12 * left, 1], dtype = float)    

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * left + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + window_size].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * left + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + window_size] #value
    return x,y

x,y = xy_Window(window_size)

x_train_set_5 = x[: math.floor(len(x) * 0.8), :]
y_train_set_5 = y[: math.floor(len(y) * 0.8), :]
x_validation_5 = x[math.floor(len(x) * 0.8): , :]
y_validation_5 = y[math.floor(len(y) * 0.8): , :]\

#%%
# trainging
dim = 18 * 9 + 1
x_train_set_length = np.shape(x_train_set)[0]
x_train = np.concatenate((np.ones([x_train_set_length, 1]), x_train_set), axis = 1).astype(float)
learning_rate_list = [1,10,50,100,200,500]
iter_time = 100000
eps = 1e-7

adagrad_loss_list = []
for lr in learning_rate_list:
    #initialize
    w = np.zeros([dim, 1])
    adagrad = np.zeros([dim, 1])
    cur_loss = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/x_train_set_length)#rmse
        cur_loss.append(loss)
        if(t%5000==0):
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set) #dim*1
        adagrad += gradient ** 2
        w = w - lr * gradient / np.sqrt(adagrad + eps)
    adagrad_loss_list.append(cur_loss)
    print(str(t) + ":" + str(loss))