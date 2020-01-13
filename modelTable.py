import numpy as np

class Model_table:
    def __init__(self):
        self.q_table = {}
        for i in range(101):
            for ii in range(101):
                for iii in range(100001)
                    self.q_table[(i, ii, iii)] = [np.random.uniform(-5, 0) for i in range(6)]