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
print(x)
print(y)
print(np.shape(x))
#%%
# Normalizing
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

np.save('training_mean.npy', mean_x)
np.save('training_std.npy', std_x)

#%%
# Training with Adagrad
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x_train = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 50
iter_time = 500000
adagrad = np.zeros([dim, 1])
eps = 1e-7
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y, 2))/471/12)#rmse
    if(t%10000==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
print(str(t) + ":" + str(loss))
np.save('weight_adagrad.npy', w)

#%%
# Training with Adam
# dim = 18 * window_size + 1
# left = 480-window_size
# w = np.zeros([dim, 1])
# x_train = np.concatenate((np.ones([12 * left, 1]), x), axis = 1).astype(float)
# learning_rate = 0.001
# iter_time = 50000
# beta_1 = 0.9
# beta_2 = 0.999
# eps = 1e-7
# m_t, v_t = np.zeros([dim,1]), np.zeros([dim,1])

# for t in range(1,iter_time,1):
#     loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y, 2))/left/12)#rmse
#     if(t%10000==0):
#         print(str(t) + ":" + str(loss))
#     g_t = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y)
#     m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
#     v_t = beta_2*v_t + (1-beta_2)*(g_t**2)	#updates the moving averages of the squared gradient
#     m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
#     v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates

#     w = w - (learning_rate*m_cap)/(np.sqrt(v_cap)+eps)	#updates the parameters
# np.save('weight_adam_final.npy',w)
# print('weight saved')
#%%
# read in testdata
testdata = pd.read_csv('test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*window_size], dtype = float)

for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
test_mean_x = np.mean(test_x, axis = 0) #18 * 9 
test_std_x = np.std(test_x, axis = 0) #18 * 9 

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if test_std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
# print(test_x)

# %%    
# w = np.load('weight_adam_final.npy')
w = np.load('weight_adam.npy')
ans_y = np.dot(test_x, w)

import csv
output_file_name = 'submit_adam'
with open(output_file_name+'.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

# %%
# testing MLE
# x_train = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
# beta = np.linalg.inv(np.dot(x_train.transpose(), x_train))
# beta = beta.dot(x_train.transpose())
# beta = beta.dot(y)

# loss = np.sqrt(np.sum(np.power(np.dot(x_train, beta) - y, 2))/471/12)
# print(loss)
# %%
