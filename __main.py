from __environment import Agent
from __testNN import testNN
from __modelTable import Model_table

import numpy as np

# parameters:
LEARNING_RATE = 0.1
DISCOUNT = 0.95
epsilon = 0.5
q_table = Model_table().q_table
episode_rewards = []

for _ in range(10):
    for episode in range(10):
        agent = Agent(param1=0, param2=0)
        episode_reward = 0
        for step in range(10):
            state = (agent.param1, agent.param2)
            if np.random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = np.random.randint(0, 4)
            # rewarding
            reward = testNN(agent.param1, agent.param2) # calling neural network
            new_state = (agent.param1, agent.param2)
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state][action]
            # Q value calculations
            new_q = (1-LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT*max_future_q)
            q_table[state][action] = new_q
            episode_reward += reward
            # TODO: break needed?
        episode_rewards.append(episode_reward)
        epsilon *= 0.9999

print(q_table)
print(episode_rewards)
