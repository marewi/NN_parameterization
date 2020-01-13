import numpy as np

class Model_table:
    def __init__(self):
        self.q_table = {}
        for i in range(10):
            for ii in range(10):
                for iii in range(1001):
                    self.q_table[(i, ii, iii)] = [np.random.uniform(1, 5) for i in range(6)]