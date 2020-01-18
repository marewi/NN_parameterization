import numpy as np
from parameters import num_epochs_stepsize, batch_size_stepsize, learning_rate_stepsize, \
    num_epochs_min, num_epochs_max, batch_size_min, batch_size_max, learning_rate_min, learning_rate_max

class Model_table:
    def __init__(self):
        amount_num_epochs = (num_epochs_max - num_epochs_min) * (1 / num_epochs_stepsize)
        amount_batch_size = (batch_size_max - batch_size_min) * (1 / batch_size_stepsize)
        amount_learning_rate = (learning_rate_max - learning_rate_min) * (1 / learning_rate_stepsize)
        self.q_table = {}
        for i in range(amount_num_epochs):
            for ii in range(amount_batch_size):
                for iii in range(amount_learning_rate):
                    self.q_table[(i/(1/num_epochs_stepsize), 
                        ii/(1/batch_size_stepsize), 
                        iii/(1/learning_rate_stepsize))] = [0 for i in range(6)]


                    # self.q_table[(i/(1/num_epochs_stepsize), 
                    #     ii/(1/batch_size_stepsize), 
                    #     iii/(1/learning_rate_stepsize))] = [np.random.uniform(9, 10) for i in range(6)]
        # print(self.q_table)