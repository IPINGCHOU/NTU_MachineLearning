#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# report 1
# read rewards
tutorial_total = np.load('tutorial_avg_total.npy')
tutorial_final = np.load('tutorial_avg_final.npy')

discount_total = np.load('discount_avg_total.npy')
discount_final = np.load('discount_avg_final.npy')
#%%
length = len(tutorial_final)
plt.plot(range(length), tutorial_total, label = 'tutorial')
plt.plot(range(length), discount_total, label = 'discount')
plt.title('tutorial vs discount rewards')
plt.legend()
plt.show()
#%%
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

ma_tutorial = moving_average(tutorial_total, 10)
ma_discount = moving_average(discount_total, 10)
length = len(ma_tutorial)
plt.plot(range(length), ma_tutorial, label = 'tutorial')
plt.plot(range(length), ma_discount, label = 'discount')
plt.title('tutorial vs discount rewards with MA = {}'.format(10))
plt.legend()
plt.show()

# %%
# report 2
# update rate
discount_total_update1 = np.load('discount_avg_total_update1.npy')
discount_total_update10 = np.load('discount_avg_total_update10.npy')

ma_discount_update1 = moving_average(discount_total_update1, 10)
ma_discount_update10 = moving_average(discount_total_update10, 10)
length = len(ma_discount)
plt.plot(range(length), ma_discount, label = 'update 5')
plt.plot(range(length), ma_discount_update1, label = 'update 1')
plt.plot(range(length), ma_discount_update10, label = 'update 10')
plt.title('discount reward with different update rate, MA = 10')
plt.legend()
plt.show()

# %%
# lr
discount_total_lr2 = np.load('discount_avg_total_lr0.01.npy')
discount_total_lr4 = np.load('discount_avg_total_lr0.0001.npy')

length = 50
plt.plot(range(50), discount_total[:length], label = 'lr 1e-03')
plt.plot(range(50), discount_total_lr2[:length], label = 'lr 1e-02')
plt.plot(range(50), discount_total_lr4[:length], label = 'lr 1e-04')
plt.legend()
plt.show()


ma_discount_lr2 = moving_average(discount_total_lr2, 10)
ma_discount_lr4 = moving_average(discount_total_lr4, 10)
length = len(ma_discount)
plt.plot(range(length), ma_discount, label = 'lr 1e-03')
plt.plot(range(length), ma_discount_lr2, label = 'lr 1e-02')
plt.plot(range(length), ma_discount_lr4, label = 'lr 1e-04')
plt.title('discount reward with different learning rate, MA = 10')
plt.legend()
plt.show()
#%%

discount_total_epoch200 = np.load('discount_avg_total_epoch200.npy')
discount_total_epoch600 = np.load('discount_avg_total_epoch600.npy')

plt.plot(range(len(discount_total_epoch600)), discount_total_epoch600, label = 'epoch 600')
plt.plot(range(len(discount_total)), discount_total, label = 'epoch 400')
plt.plot(range(len(discount_total_epoch200)), discount_total_epoch200, label = 'epoch 200')
plt.legend()
plt.show()

# %%
# report a2c
a2c_total = np.load('a2c_avg_total_epoch400.npy')
a2c_final = np.load('a2c_avg_final_epoch400.npy')

length = len(discount_total)
plt.plot(range(length), discount_total, label = 'PG with reward discount, epoch = 400')
plt.plot(range(length), a2c_total, label = 'A2C , epoch = 400')
plt.title('A2C vs PG, total rewards for each episodes')
plt.legend()
plt.show()

# %%
ma_discount = moving_average(discount_total, 10)
ma_a2c = moving_average(a2c_total, 10)

length = len(ma_discount)
plt.plot(range(length), ma_discount, label = 'PG with reward discount, epoch = 400')
plt.plot(range(length), ma_a2c, label = 'A2C , epoch = 400')
plt.title('A2C vs PG, total rewards for each episodes, MA = 10')
plt.legend()
plt.show()
#%%
a2c_total_epoch600 = np.load('a2c_avg_total_epoch600_lr0.001.npy')
ma_discount_total_600 = moving_average(discount_total_epoch600, 10)
ma_a2c_total_600 = moving_average(a2c_total_epoch600, 10)
length = len(ma_discount_total_600)
plt.plot(range(length), ma_discount_total_600, label = 'PG with reward discount, epoch = 600')
plt.plot(range(length), ma_a2c_total_600, label = 'A2C , epoch = 600')
plt.title('A2C vs PG, total rewards for each episodes, MA = 10')
plt.legend()
plt.show()


# %%
# final reward a2c vs discount PG
a2c_final = np.load('a2c_avg_final_epoch400.npy')

length = len(discount_final)
plt.plot(range(length), discount_final, label = 'PG with reward discount')
plt.plot(range(length), a2c_final, label = 'A2C')
plt.title('A2C vs PG, final rewards for each episodes')
plt.legend()
plt.show()
#%%
