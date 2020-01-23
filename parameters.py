################################
### ENVIRONMENT & MODEL PARAMETERS
# how big is the step the agent can take for NN learning paramters:
num_epochs_stepsize = 1
batch_size_stepsize = 1
learning_rate_stepsize = 0.001

# barriers of parameters
num_epochs_min = 15
num_epochs_max = 30

batch_size_min = 15
batch_size_max = 30

learning_rate_min = 0
learning_rate_max = 0.1


################################
### LEARNING PARAMETERS
# amount of episodes
episodes = 1

# amount of steps (per episode)
steps = 1

# learning rate of RL
LR = 0.1

# discount factor of RL
DISCOUNT = 0.95

# epsilon of RL
epsilon = 0.5

# epsilon decay of RL
EPSILON_DECAY = 0.9999