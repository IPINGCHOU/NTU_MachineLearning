#%%
%%capture
from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

%matplotlib inline
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
#%%
%%capture
import gym
env = gym.make('LunarLander-v2')

#%%
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
    
    def forward(self, x):
        x = torch.FloatTensor(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return F.softmax(x, dim = -1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

class A2C:
    def __init__(self, Actor, Critic, Memory):
        self.actor = Actor
        self.critic = Critic
        self.memory = Memory
        self.optim_actor = optim.SGD(self.actor.parameters(), lr=LEARNING_RATE)
        self.optim_critic = optim.SGD(self.critic.parameters(), lr = LEARNING_RATE)
        self.gamma = GAMMA

    def sample(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist

    def update(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory),1))
        
        for i, (_,_, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma * q_val * (1-done)
            q_vals[len(self.memory) -1 - i] = q_val
        
        adv = torch.Tensor(q_vals) - values
        # update model
        # critic
        critic_loss = adv.pow(2).mean()
        agent.optim_critic.zero_grad()
        critic_loss.backward()
        agent.optim_critic.step()
        # actor
        actor_loss = (-torch.stack(self.memory.log_probs) * adv.detach()).mean()
        agent.optim_actor.zero_grad()
        actor_loss.backward()
        agent.optim_actor.step()

    
#%%
# training

EPISODE_PER_BATCH = 1  # 每蒐集 n 個 episodes 更新一次 agent
NUM_BATCH = 600        # 總共更新 400 次
# DISCOUNT = True
GAMMA = 0.99           # discount factor
LEARNING_RATE = 0.001
MAX_STEP = 200

actor = Actor()
critic = Critic()
memory = Memory()
agent = A2C(actor, critic, memory)
agent.actor.train()  # 訓練前，先確保 network 處在 training 模式
agent.critic.train()

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    rewards = []
    total_rewards, final_rewards = [], []

    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        
        state = env.reset()
        total_reward, total_step = 0, 0

        while True:

            action, dist = agent.sample(state)
            next_state, reward, done, _ = env.step(action.detach().data.numpy())
            # adv = reward + (1-done) * agent.gamma * agent.critic(next_state) - agent.critic(state)

            total_reward += reward
            total_step += 1
            agent.memory.add(dist.log_prob(action), critic(state), reward, done)
            state = next_state


            if done or (total_step % MAX_STEP == 0):
                last_q_val = critic(next_state).detach().data.numpy()
                agent.update(last_q_val)
                agent.memory.clear()

            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                rewards.append(np.full(total_step, total_reward))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                break

    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")



#%%
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()
#%%
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()
#%%
n1 = 'a2c_avg_total_epoch600_lr0.001'
n2 = 'a2c_avg_final_epoch600_lr0.001'
np.save(n1, avg_total_rewards)
np.save(n2, avg_final_rewards)

#%%
agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

state = env.reset()

img = plt.imshow(env.render(mode='rgb_array'))

total_reward = 0

done = False
while not done:
    action, _ = agent.sample(state)
    state, reward, done, _ = env.step(action)

    total_reward += reward

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)