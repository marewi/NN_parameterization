import numpy as np
from parameters import num_epochs_stepsize, batch_size_stepsize, learning_rate_stepsize

class Model_table:
    def __init__(self):
        self.q_table = {}
        for i in range(11):
            for ii in range(11):
                for iii in range(1001):
                    self.q_table[(i/(1/num_epochs_stepsize), 
                        ii/(1/batch_size_stepsize), 
                        iii/(1/learning_rate_stepsize))] = [0 for i in range(6)]
                    # self.q_table[(i/(1/num_epochs_stepsize), 
                    #     ii/(1/batch_size_stepsize), 
                    #     iii/(1/learning_rate_stepsize))] = [np.random.uniform(9, 10) for i in range(6)]
        # print(self.q_table)