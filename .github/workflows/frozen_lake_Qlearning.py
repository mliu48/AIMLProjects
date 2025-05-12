import numpy as np
import gym
import random

# 创建FrozenLake环境
env = gym.make("FrozenLake-v1", is_slippery=True)

# Q表初始化
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

# 超参数
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 6000

# 训练 Q-learning 代理
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        
        next_state, reward, done, _, _ = env.step(action)
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        state = next_state
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()

# 运行智能体并渲染最终策略
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
state = env.reset()[0]
done = False

print("Trained Policy Execution")
while not done:
    action = np.argmax(q_table[state, :])
    state, _, done, _, _ = env.step(action)
    env.render()

env.close()
