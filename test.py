from parameters import num_epochs_stepsize, batch_size_stepsize, learning_rate_stepsize
import operator
import numpy as np

tt = {}

for i in range(3):
            for ii in range(1):
                for iii in range(1):
                    tt[(i,ii,iii)] = [np.random.randint(1,10) for i in range(6)]

print(tt)
print("-------------------------------")
for key in tt:
    print(np.argmax(tt[key]))