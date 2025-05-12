import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')
action_space = env.action_space.n
obs_space = env.observation_space.n

Q = np.zeros((obs_space, action_space))  # 将Q值初始化为0

epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.999  
episodes = 500  
lr = 0.1  
gamma = 0.99

def epsilon_greedy(state,epsilon):
    if np.random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state,:])
    
rewards = []

for epoch in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = epsilon_greedy(state,epsilon)
        next_state,reward,done,_,_ = env.step(action)
        next_action = epsilon_greedy(next_state,epsilon)
        Q[state, action] = Q[state, action] + lr * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        state = next_state
        total_reward+=reward
    #epsilon = max(epsilon_min,epsilon * epsilon_decay)
    rewards.append(total_reward)
    print(f'Epoch: {epoch}, reward: {total_reward}')
env.close()

env = gym.make('CliffWalking-v0',render_mode = 'human')
state = env.reset()[0]
done = False

while not done:
    env.render()
    action = np.argmax(Q[state,:])
    next_state,reward,done,_,_ = env.step(action)
    state = next_state
env.close()

plt.plot(rewards)
plt.xlabel('epochs')
plt.ylabel('rewards')
plt.show()