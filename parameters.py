# size of grid environment
SIZE = 128

# this is per training data (picture) -> 50000 per pic
EPISODES = 1 #10000

# how often is the agent allowed to try to reach the goal?
steps = 200

# punishing the agent for each step
MOVE_PENALTY = 1

# rewarding the agent when it reaches the goal
GOAL_REWARD = 100

epsilon = 0.5

# every episode will be epsilon*EPISODE_DECAY
EPISODE_DECAY = 0.9999

# how often to play through env visually
SHOW_EVERY = 10

# aka step size: weighted the strength of change of q-value
LEARNING_RATE = 0.1

# aka visual range: weighted the influence of (max-) future q-value
DISCOUNT = 0.95