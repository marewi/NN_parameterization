import numpy as np

class Model_table:
    def __init__(self):
        self.q_table = {}
        start_q_table = None  # here can be inserted a existing file
        if start_q_table is None:
            # i,ii = {0;127}
            for i in range(11):
                for ii in range(11):
                    self.q_table[(i, ii)] = [np.random.uniform(-5, 0)
                                             for i in range(4)]