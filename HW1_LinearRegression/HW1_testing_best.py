# applying Adagrad
# learning_rate = 50
# iter_time = 500000

import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

w = np.load('weight_adam.npy')
print('Applying Adam')
mean_x = np.load('training_mean.npy')
std_x = np.load('training_std.npy')

filename = sys.argv[0]
input_route = sys.argv[1]
output_route = sys.argv[2]

window_size = 9
testdata = pd.read_csv(input_route, header = None, encoding = 'big5')
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

ans_y = np.dot(test_x, w)

import csv
with open(output_route, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

print('ans file generated')