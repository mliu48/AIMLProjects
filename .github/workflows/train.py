import gym
import custom_env  # This will ensure the custom environment is registered
import gym
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# Load the custom environment
env = gym.make('CustomEnv-v35')


env.continuous = False
env.gravity = 3.71
env.enable_wind = True
env.wind_power = 15.0
env.turbulence_power = 1.5


obs_space = env.observation_space
action_space = env.action_space
print(obs_space.shape, action_space.n)

input_dim = obs_space.shape[0]
output_dim = action_space.n

env.reset()
env.action_space.sample()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)

# Hyperparameters
gamma = 0.99
batch_size = 64
replay_buffer_size = 100000
learning_rate = 0.001
target_update_frequency = 10

model = DQN(input_dim, output_dim)

target_model = DQN(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())
target_model.eval()


criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

replay_buffer = deque(maxlen=replay_buffer_size)

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:  # Explore
        return env.action_space.sample()
    else:  # Exploit
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return model(state).argmax().item()

def train_step(model, target_model, optimizer, criterion, batch):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def sample_replay_buffer(batch_size):
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

rwd = []

def train(model, target_model, criterion, optimizer, episodes, epsilon, min_epsilon, epsilon_decay):

    num_success = 0


    for episode in range(episodes):
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:

            action = choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                batch = sample_replay_buffer(batch_size)
                train_step(model, target_model, optimizer, criterion, batch)
            
        
        with open('rewards.txt', 'a') as f:
            f.write(str(total_reward)+'\n')
        rwd.append(total_reward)
        
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'model_episode{episode}.pth')
            
        if episode % 50 == 0 and episode<50:
                print(f'Episode: {episode}, Total Reward: {total_reward}') 
        elif episode % 50 == 0 and episode>=50:
            print(f'Episode: {episode},  Average Reward: {sum(rwd[-50:])/50}')
        
        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())
        
        r = []
        with open('rewards.txt', 'r') as file:
            for line in file:
                r.append(float(line.strip()))
    
        if all(val > 200 for val in r[-7:]):
            print(f'Solved at episode {episode}')
            num_success += 1
            if num_success == 3:
                epsilon = 0.01 * epsilon
            elif num_success <= 2:
                epsilon *= 0.55
        else:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

train(model, target_model, criterion, optimizer, episodes=10000, min_epsilon=0.01, epsilon=1.0, epsilon_decay=0.995)



