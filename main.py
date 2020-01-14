import numpy as np

from environment import Agent
from modelTable import Model_table
from neural_network.main import train
from parameters import LR, DISCOUNT, epsilon, EPSILON_DECAY, episodes, steps


print("Creating RL model...")
q_table = Model_table().q_table
episode_rewards = []

print("Starting to train RL model...")
for episode in range(episodes):
    agent = Agent(num_epochs=1, batch_size=1, learning_rate=0)
    episode_reward = 0
    for step in range(steps):
        state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
        if np.random.random() > epsilon:
            action = np.argmin(q_table[state])
        else:
            action = np.random.randint(0, 6)
        agent.action(action) # take the action
        # rewarding
        reward = train(agent.num_epochs, agent.batch_size, agent.learning_rate) # calling neural network
        print(f"reward = {reward}")
        new_state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
        min_future_q = np.min(q_table[new_state])
        current_q = q_table[state][action]
        # Q value calculations
        new_q = (1-LR) * current_q + LR * (reward + DISCOUNT*min_future_q)
        q_table[state][action] = new_q
        episode_reward += reward
        # TODO: break needed?
    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

# print(q_table)
print(episode_rewards)
