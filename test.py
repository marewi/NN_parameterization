import math
import numpy as np
from parameters import *
from neural_network.main import train
# from modelTable import Model_table

# try:
#     test_reward = train(5, 10, 0.001)
# except Exception as e:
#     print(str(e))

# # if test_reward == 'nan':
# #     print("Error caught")

# print(test_reward)

# q_table = Model_table().q_table
# experienced_rewards = {}
# for key in q_table.keys():
#     experienced_rewards[key] = 0

# print(experienced_rewards)

# print(experienced_rewards[(10, 10, 0.1)])

       expert = train(20, 16, 0.001) -> 0.1523854398727417
training_loss = train(26, 18, 0.015) -> 0.01136340035591843

print(manually)




# #####################

# learning_rate_start = np.random.randint(learning_rate_min/learning_rate_stepsize, \
#             learning_rate_max/learning_rate_stepsize)*learning_rate_stepsize
# lr_decimal_place = str(abs(int(math.log10(learning_rate_stepsize)))) # decimal place of lr_stepsize
# lr_decimal_place_str = "{0:." + lr_decimal_place + "f}"  # "{0:.2f}" # building formating string
# learning_rate_start = float(lr_decimal_place_str.format(learning_rate_start))

# print(learning_rate_start)


# a = 0.036000000000000004
# print(a)

# b = float("{0:.3f}".format(a))
# print(b)
# print(type(b))