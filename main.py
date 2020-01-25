import datetime
import math
import operator
import time

import numpy as np
from termcolor import colored

from environment import Agent
from lib.minLoss import getMinLoss
from lib.writeXLSX import writeXLSX
from modelTable import Model_table
from neural_network.main import train
from parameters import *

EPSILON = epsilon
start_time = time.time()
print("Creating RL model...")
q_table = Model_table().q_table
experienced_rewards = Model_table().experienced_rewards
# episode_rewards = []
min_losses = []
barrier_counter = 0
num_NN_train = 0

print("Starting to train RL model...")
for episode in range(episodes):
    # start agent in random state
    num_epochs_start = np.random.randint(num_epochs_min, num_epochs_max)
    batch_size_start = np.random.randint(batch_size_min, batch_size_max)
    learning_rate_start = np.random.randint(learning_rate_min/learning_rate_stepsize, \
            learning_rate_max/learning_rate_stepsize)*learning_rate_stepsize
    lr_decimal_place = str(abs(int(math.log10(learning_rate_stepsize)))) # decimal place of lr_stepsize
    lr_decimal_place_str = "{0:." + lr_decimal_place + "f}"  # "{0:.2f}" # building formating string
    learning_rate_start = float(lr_decimal_place_str.format(learning_rate_start))
    # create agent in random start
    agent = Agent(num_epochs=num_epochs_start, \
        batch_size=batch_size_start, \
        learning_rate=learning_rate_start)
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
            barrier_counter +=1
            print(colored("barrier", 'red'))
            continue
        # rewarding
        print(f"new state: {agent.num_epochs} | {agent.batch_size} | {agent.learning_rate}")
        if experienced_rewards[state] > 0:
                reward = experienced_rewards[state]
                print(f"reward was experienced yet & was gestored")
        else:
            try:
                loss = train(agent.num_epochs, agent.batch_size, agent.learning_rate) # calling neural network
                if math.isnan(loss) == True: #loss == 'nan' or loss == 'NaN':
                    reward = 0
                    print(f"loss was to high: {loss} -> reward={reward}")
                else:
                    reward = 128 - loss
                    if reward < 0:
                        reward = 0
                num_NN_train += 1
            except Exception as e: # e.g. when "can't allocate memory"
                print(e)
                continue
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
        # speedup: save seen reward
        experienced_rewards[state] = reward
        # save successes
        current_min_loss = getMinLoss(experienced_rewards)
        min_losses.append(current_min_loss)
    # episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

print(f"---------------------------TRAINING IS DONE----------------------------")
# print(q_table)
# print(episode_rewards)

### identify best parameter set
max_v_value = 0
max_v_key = (0,0,0)
for key in q_table:
    v_value = np.max(q_table[key])
    if v_value > max_v_value:
        max_v_value = v_value
        max_v_key = key

final_min_loss_key, final_min_loss_value = getMinLoss(experienced_rewards)

overall_time = time.time() - start_time

writeXLSX(min_losses)

dt = datetime.datetime.now()

file = open("results.txt", "w")
file.write(f"num_epochs_stepsize: {num_epochs_stepsize}\nbatch_size_stepsize: {batch_size_stepsize}\nlearning_rate_stepsize: {learning_rate_stepsize}\nnum_epochs_min: {num_epochs_min}\nnum_epochs_max: {num_epochs_max}\nbatch_size_min: {batch_size_min}\nbatch_size_max: {batch_size_max}\nlearning_rate_min: {learning_rate_min}\nlearning_rate_max: {learning_rate_max}\n\n")
file.write(f"RL episodes: {episodes}\nRL steps: {steps}\nRL LR: {LR}\nRL discount factor: {DISCOUNT}\nRL epsilon: {EPSILON}\nRL epsilon decay: {EPSILON_DECAY}\n\n")
file.write(f"amount of barrier bumps: {barrier_counter}\namount of NN trainings: {num_NN_train}\noverall exec time: {overall_time}\noverall max V value: {max_v_value}\noverall min loss value: {final_min_loss_value}\noverall best parameter set: {final_min_loss_key}\n\n")
file.write(f"datetime: {dt}")
file.close()

# print(f"min_losses: {min_losses}")
# print(f"amount of barrier bumps: {barrier_counter}")
# print(f"amount of NN trainings: {num_NN_train}")
# print(colored(f"overall max V value: {max_v_value}", 'cyan'))
# print(colored(f"overall min loss value: {final_min_loss_value}", 'cyan'))
# print(colored(f"overall best parameter set: {final_min_loss_key}", 'cyan'))
