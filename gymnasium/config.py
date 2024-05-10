import os

#**** Environment ********
ENV = "Humanoid-v4"
PATH_SAVE = "./new_models/"
PATH_LOAD = None

#******** Network ********

ACTOR_HIDDEN = [1024, 512]
CRITIC_HIDDEN = [1024, 512]

LOG_STD_MIN = -20 # exp(-10) = 4.540e-05
LOG_STD_MAX = 2 # exp(2) = 7.389
EPSILON = 1e-6

#**** Replay Buffer ******
MIN_SIZE_BUFFER = 100
BUFFER_CAPACITY = 100000000
BATCH_SIZE = 256


#******* Training ********
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 0.0005
CRITIC_LR = 0.0005
REWARD_SCALE = 2
THETA = 0.15
DT = 1e-1


