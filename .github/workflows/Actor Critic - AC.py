import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('CartPole-v1')

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=24):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=24):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 超参数
episodes = 300
gamma = 0.99  # 折扣因子
lr_actor = 0.001  # 策略网络学习率
lr_critic = 0.002  # 价值网络学习率
batch_size = 64
replay_buffer = deque(maxlen=10000)
plot_reward = []

# 初始化网络
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

actor = Actor(in_dim, out_dim)
critic = Critic(in_dim)
optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

for epoch in range(episodes):
    state = env.reset()[0]
    
    state = torch.FloatTensor(state).unsqueeze(0)
    
    done = False
    total_reward = 0
    step = 0
    while not done and step < 2000:
        step += 1
        # 选择动作
        with torch.no_grad():
            probs = actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        total_reward += reward
        
        # 存储过渡
        #print(state.squeeze(0).shape,next_state.squeeze(0).shape)
        replay_buffer.append((state.squeeze(0), action, reward, next_state.squeeze(0), done))
        state = next_state
        
        # 更新网络
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            state_batch = torch.stack(state_batch)
            
            action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
            next_state_batch = torch.stack(next_state_batch)
            done_batch = torch.FloatTensor(done_batch).unsqueeze(1)
            
            # 计算TD目标
            with torch.no_grad():
                next_value = critic(next_state_batch)
                td_target = reward_batch + gamma * next_value * (1 - done_batch)
            
            # 计算当前值
            current_value = critic(state_batch)
            
            # 计算TD误差
            td_error = td_target - current_value
            
            # 更新策略网络
            probs = actor(state_batch)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(action_batch)
            actor_loss = -torch.mean(log_probs * td_error.detach())
            
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            # 更新价值网络
            critic_loss = F.mse_loss(current_value, td_target.detach())
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
    
    plot_reward.append(total_reward)
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Total Reward: {total_reward}')

# 保存模型
torch.save(actor.state_dict(), 'actor.pth')
torch.save(critic.state_dict(), 'critic.pth')

# 可视化训练过程
plt.plot(plot_reward)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Actor-Critic Training on CartPole')
plt.show()

# 测试模型
env = gym.make('CartPole-v1', render_mode='human')
state = env.reset()[0]
state = torch.FloatTensor(state).unsqueeze(0)
done = False
loaded_actor = Actor(in_dim, out_dim)
loaded_actor.load_state_dict(torch.load("actor.pth"))
loaded_actor.eval()
step = 0
while not done and step < 2000:
    step += 1
    env.render()
    with torch.no_grad():
        probs = loaded_actor(state)
        action = torch.argmax(probs).item()
    next_state, reward, done, _, _ = env.step(action)
    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    state = next_state