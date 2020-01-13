import numpy as np

from environment import Agent
from modelTable import Model_table
# from testNN import testNN
from neural_network.main import train


### parameters:
LR = 0.1
DISCOUNT = 0.95
epsilon = 0.5
q_table = Model_table().q_table
episode_rewards = []

for _ in range(10):
    for episode in range(10):
        agent = Agent(num_epochs=0, batch_size=0, learning_rate=0)
        episode_reward = 0
        for step in range(10):
            state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
            if np.random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = np.random.randint(0, 4)
            # rewarding
            reward = train(agent.num_epochs, agent.batch_size, agent.learning_rate) # calling neural network
            new_state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state][action]
            # Q value calculations
            new_q = (1-LR) * current_q + LR * (reward + DISCOUNT*max_future_q)
            q_table[state][action] = new_q
            episode_reward += reward
            # TODO: break needed?
        episode_rewards.append(episode_reward)
        epsilon *= 0.9999

print(q_table)
print(episode_rewards)
