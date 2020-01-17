import numpy as np
import operator
from termcolor import colored

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
        print(f"---------------------------------{episode}, {step}")
        state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            print(colored("random action will be taken", 'blue'))
            action = np.random.randint(0, 6)
        print(f"current state: {agent.num_epochs} | {agent.batch_size} | {agent.learning_rate}")
        try:
            agent.action(action) # take the action
        except: # if agent runs against barrier of environment
            print(colored("barrier", 'red'))
            continue
        # rewarding
        print(f"new state: {agent.num_epochs} | {agent.batch_size} | {agent.learning_rate}")
        reward = 128 - train(agent.num_epochs, agent.batch_size, agent.learning_rate) # calling neural network
        print(colored(f"reward = {reward}", 'green'))
        new_state = (agent.num_epochs, agent.batch_size, agent.learning_rate)
        max_future_q = np.max(q_table[new_state])
        current_q = q_table[state][action]
        # Q value calculations
        new_q = (1-LR) * current_q + LR * (reward + DISCOUNT*max_future_q)
        print(f"old q value: {current_q}")
        print(f"new q value: {new_q}")
        q_table[state][action] = new_q
        episode_reward += reward
    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

print(f"---------------------------TRAINING IS DONE----------------------------")
# print(q_table)
# print(episode_rewards)

### identify best parameter set
max_v_value = 0
max_v_value_key = (0,0,0)
for key in q_table:
    v_value = np.max(q_table[key])
    if v_value > max_v_value:
        max_v_value = v_value
        max_v_value_key = key

print(colored(f"overall max V value: {max_v_value}", 'cyan'))
print(colored(f"overall best parameter set: {max_v_value_key}", 'cyan'))
