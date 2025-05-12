import numpy as np
import gym

env = gym.make('FrozenLake-v1',is_slippery=False)
num_actions = env.action_space.n
num_states = env.observation_space.n


Q = np.zeros((num_states,num_actions))

epochs = 1000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
lr = 0.8

def epsilon_greedy(state,epsilon):
    if np.random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state,:])
    return action

for epoch in range(epochs):
    #init
    state = env.reset()[0]
    done = False
    action = epsilon_greedy(state, epsilon)
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(next_state,epsilon)
        Q[state,action] += lr*(reward + gamma*(Q[next_state,next_action] - Q[state,action]))
        state = next_state
        action = next_action

    epsilon = max(epsilon_min,epsilon_decay*epsilon)
    if epoch%50 == 0:
        print(f'reward: {reward}')
env.close()

env = gym.make('FrozenLake-v1',render_mode = 'human',is_slippery=False)
state = env.reset()[0]
done = False

while not done:
    action = np.argmax(Q[state,:])
    next_state, reward, done, _, _ = env.step(action)
    env.render()
env.close()