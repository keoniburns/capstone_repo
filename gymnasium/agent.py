from config import *
from replay_buffer import *
from networks import *
import sys
sys.path.append("../src")
import tensorflow as tf
from tensorflow.keras import optimizers as opt
import numpy as np
import random
import time

class Agent:

    def __init__(self, env, path_save=PATH_SAVE,
                 path_load=PATH_LOAD, actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR,
                 gamma=GAMMA, tau=TAU,
                 reward_scale=REWARD_SCALE):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load

        self.actor = Actor(actions_dim=self.actions_dim, name='actor', upper_bound=env.action_space.high)
        self.critic0 = Critic(name='critic_0')
        self.critic1 = Critic(name='critic_1')
        self.critic_value = CriticValue(name='value')
        self.critic_target_value = CriticValue(name='target_value')

        self.actor.compile(optimizer=opt.Adam(learning_rate=self.actor_lr))
        self.critic0.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic1.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_target_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))

        self.reward_scale = reward_scale
        # potentailly incorrect .get_weights() instead of just .weights
        self.critic_value.set_weights(self.critic_target_value.get_weights())


    def update_target_networks(self, tau):
        critic_value_weights = self.critic_value.get_weights()  #get_weights() instead of just .weights
        critic_target_value_weights = self.critic_target_value.get_weights()
        for i in range(len(critic_target_value_weights)):
            critic_target_value_weights[i] = tau * critic_value_weights[i] + (1 - tau) * critic_target_value_weights[i]

        self.critic_target_value.set_weights(critic_target_value_weights)

    def save():
