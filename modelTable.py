import numpy as np
from parameters import num_epochs_stepsize, batch_size_stepsize, learning_rate_stepsize, \
    num_epochs_min, num_epochs_max, batch_size_min, batch_size_max, learning_rate_min, learning_rate_max

class Model_table:
    def __init__(self):
        amount_num_epochs = int((num_epochs_max - num_epochs_min + num_epochs_stepsize) * (1 / num_epochs_stepsize))
        amount_batch_size = int((batch_size_max - batch_size_min + batch_size_stepsize) * (1 / batch_size_stepsize))
        amount_learning_rate = int((learning_rate_max - learning_rate_min + learning_rate_stepsize) * (1 / learning_rate_stepsize))
        self.q_table = {}
        for i in range(amount_num_epochs):
            i += num_epochs_min # dont start with 0
            for ii in range(amount_batch_size):
                ii += batch_size_min # dont start with 0
                for iii in range(amount_learning_rate):
                    iii += learning_rate_min # dont start with 0
                    self.q_table[(i/(1/num_epochs_stepsize), 
                        ii/(1/batch_size_stepsize), 
                        iii/(1/learning_rate_stepsize))] = [0 for i in range(6)]
        # print(amount_num_epochs)
        # print(amount_batch_size)
        # print(amount_learning_rate)
        self.experienced_rewards = {}
        for key in self.q_table.keys():
            self.experienced_rewards[key] = 0
