import numpy as np
import gym


env = gym.make("FrozenLake-v1")

num_states = env.observation_space.n
states = range(num_states)
num_actions = env.action_space.n
actions = range(num_actions)

V = np.zeros(num_states)
policy = np.ones((num_states,num_actions))/num_actions

max_iter = 5000
gamma = 0.99
theta = 1e-4

for epoch in range(max_iter):
    #policy evaluation
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = sum([policy[s][a] * sum([prob * (reward + gamma * V[next_state]) 
                                            for prob, next_state, reward, done in env.P[s][a]]) 
                        for a in range(num_actions)])
            delta = max(delta, abs(V[s] - v))
            
        if delta < theta:
            break
    
        if delta < theta:
            break
    #policy improvement
    is_stable = True
    for s in states:
        current_action = np.argmax(policy[s])
        Q = np.array([sum([prob*(reward+gamma*V[next_state])
                             for prob,next_state,reward,done in env.P[s][a]]) 
                             for a in actions])
        best_action = np.argmax(Q)
        policy[s] = np.eye(num_actions)[best_action]
        if best_action != current_action:
            is_stable = False
        
    if is_stable == True:
        break

for s in states:
    print(f'for {s}, policy is:{policy[s]}')
