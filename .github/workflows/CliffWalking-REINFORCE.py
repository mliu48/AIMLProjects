import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import numpy as np

def one_hot_encode(state, num_states=48):
    one_hot = np.zeros(num_states)
    one_hot[state] = 1
    return one_hot


class PolicyNetwork(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim = 128):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim = -1)
        return x

    
class REINFORCE:
    def __init__(self,policy,gamma=0.995,lr=0.2):
        self.optimizer = optim.Adam(policy.parameters(),lr=lr)
        self.policy = policy
        self.gamma = gamma
    def choose_action(self,state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dict = torch.distributions.Categorical(probs)
        action = dict.sample()
        log_prob = dict.log_prob(action)
        return action.item() ,log_prob
    def update(self, rewards, log_probs):
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0,G)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        loss = []

        for prob,dr in zip(log_probs,discounted_rewards):
            loss.append(-prob * dr)
        policy_loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

rewards_plot = []

def train(env_name,episodes):
    env = gym.make(env_name)
    obs_space = env.observation_space.n
    action_space = env.action_space.n
    policy = PolicyNetwork(obs_space,action_space)
    agent = REINFORCE(policy)

    for epoch in range(episodes):
        total_reward = 0
        state = env.reset()[0]
        state = torch.FloatTensor(one_hot_encode(state))  # 修正 `state` 传递方式

        done = False

        log_probs = []
        rewards = []
        while not done:
            action,log_prob = agent.choose_action(state)
            next_state,reward,done,_,_ = env.step(action)
            next_state = torch.FloatTensor(one_hot_encode(next_state))  # 修正 `next_state`

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward+=reward
        rewards_plot.append(total_reward)
        agent.update(rewards,log_probs)

        print(f'Epoch: {epoch}, Reward: {total_reward}')
    return policy
    env.close()

def test(env_name):
    env = gym.make(env_name,render_mode = 'human')
    state = env.reset()[0]
    state = torch.FloatTensor(one_hot_encode(state))  # 修正 `state`
    done = False
    obs_space = env.observation_space.n
    action_space = env.action_space.n   
    policy = PolicyNetwork(obs_space,action_space)
    policy.load_state_dict(torch.load('CliffWalking-REINFORCE.pth'))
    agent = REINFORCE(policy)
    while not done:
        env.render()
        action,_ = agent.choose_action(state)
        next_state,r,done,_,_ = env.step(action)
        state = torch.FloatTensor(one_hot_encode(next_state))  # 修正 `state`
    env.close()

def save_model(policy,name='CliffWalking-REINFORCE.pth'):
    torch.save(policy.state_dict(),name)

if __name__ == '__main__':
    policy = train('CliffWalking-v0',5000)
    save_model(policy)
    test('CliffWalking-v0')