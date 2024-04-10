import os

#**** Environment ********
ENV = "Humanoid-v4"

#******** Network ********

ACTOR_HIDDEN = [1024, 512]
CRITIC_HIDDEN = [1024, 512]

LOG_STD_MIN = -20 # exp(-10) = 4.540e-05
LOG_STD_MAX = 2 # exp(2) = 7.389
EPSILON = 1e-6


