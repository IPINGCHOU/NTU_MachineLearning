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
window_size = 5
def xy_Window(window_size = 9):
    left = 480-window_size
    x = np.empty([12 * left, 18 * window_size], dtype = float)
    y = np.empty([12 * left, 1], dtype = float)    

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > (24-window_size-1):
                    continue
                x[month * left + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + window_size].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * left + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + window_size] #value
    return x,y

x,y = xy_Window(window_size)
print(x)
print(y)

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
#========================================
# Problem 1
# Training with Adagrad
# Testing different learning rate
dim = 18 * 9 + 1

x_train_set_length = np.shape(x_train_set)[0]
x_train = np.concatenate((np.ones([x_train_set_length, 1]), x_train_set), axis = 1).astype(float)
learning_rate_list = [0.1, 1,10,100,200,1000]
iter_time = 1000
eps = 1e-7

adagrad_loss_list = []
for lr in learning_rate_list:
    #initialize
    w = np.zeros([dim, 1])
    adagrad = np.zeros([dim, 1])
    cur_loss = []
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/x_train_set_length)#rmse
        if(t%50==0):
            # print(str(t) + ":" + str(loss))
            cur_loss.append(loss)
        gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set) #dim*1
        adagrad += gradient ** 2
        w = w - lr * gradient / np.sqrt(adagrad + eps)
    adagrad_loss_list.append(cur_loss)
    print(str(t) + ":" + str(loss))
# all loss by adagrad were stored in adagrad_loss_list
# tested lr : [1,10,50,100,200,500]

#%%
import matplotlib.pyplot as plt
plt.title('lr from 1 to 1000, by Adagrad')
plt.ylabel('RMSE')
plt.xlabel('iterations (x 50)')
plot_x = np.arange(np.shape(adagrad_loss_list)[1])
for lr in range(np.shape(adagrad_loss_list)[0]):
    plt.plot(plot_x, adagrad_loss_list[lr], label = learning_rate_list[lr])
plt.legend()
plt.show()
plt.savefig('adagrad_lr_rmse.png')

#%%
#==================================
# Problem 2
# take only 5 datas instead of 9 datas in an iteration
# a moving window function for cuting datas
window_size = 9
x,y = xy_Window(window_size)
print(x)
print(y)

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

#%%
# training
dim = 18 * window_size + 1

w = np.zeros([dim, 1])
x_train = np.concatenate((np.ones([np.shape(x_train_set)[0], 1]), x_train_set), axis = 1).astype(float)
learning_rate = 0.001
iter_time = 50000
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-7
m_t, v_t = np.zeros([dim,1]), np.zeros([dim,1])

for t in range(1,iter_time,1):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/np.shape(x_train)[0])#rmse
    if(t%5000==0):
        print(str(t) + ":" + str(loss))
    g_t = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set)
    m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t**2)	#updates the moving averages of the squared gradient
    m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates

    w = w - (learning_rate*m_cap)/(np.sqrt(v_cap)+eps)	#updates the parameters

x_validation = np.concatenate((np.ones([np.shape(x_validation)[0], 1]), x_validation), axis = 1).astype(float)

ans_y_validation_loss =  np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/np.shape(y_validation)[0])
ans_y_train_set_loss =  np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/np.shape(y_train_set)[0])

print('Training loss: ' + str(ans_y_train_set_loss))
print('Validset loss: ' + str(ans_y_validation_loss))

# %%
#=================================
# Problem 3
# set a window with size = 10 datas, 1~9th for x and 10th for y
window_size = 9
def xy_Window(window_size = 9):
    left = 480-window_size
    x = np.empty([12 * left, 1 * window_size], dtype = float)
    y = np.empty([12 * left, 1], dtype = float)    

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > (24-window_size-1):
                    continue
                x[month * left + day * 24 + hour, :] = month_data[month][9,day * 24 + hour : day * 24 + hour + window_size].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                y[month * left + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + window_size] #value
    return x,y

x,y = xy_Window(window_size)
print(x)
print(y)

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
# training
dim = 1 * window_size + 1

w = np.zeros([dim, 1])
x_train = np.concatenate((np.ones([np.shape(x_train_set)[0], 1]), x_train_set), axis = 1).astype(float)
learning_rate = 0.001
iter_time = 50000
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-7
m_t, v_t = np.zeros([dim,1]), np.zeros([dim,1])

for t in range(1,iter_time,1):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/np.shape(x_train)[0])#rmse
    if(t%5000==0):
        print(str(t) + ":" + str(loss))
    g_t = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set)
    m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t**2)	#updates the moving averages of the squared gradient
    m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates

    w = w - (learning_rate*m_cap)/(np.sqrt(v_cap)+eps)	#updates the parameters

x_validation = np.concatenate((np.ones([np.shape(x_validation)[0], 1]), x_validation), axis = 1).astype(float)

ans_y_validation_loss =  np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/np.shape(y_validation)[0])
ans_y_train_set_loss =  np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2))/np.shape(y_train_set)[0])

print('Training loss: ' + str(ans_y_train_set_loss))
print('Validset loss: ' + str(ans_y_validation_loss))

