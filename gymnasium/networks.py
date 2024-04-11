import sys
sys.path.append('../gymnasium')
from config import *
from replay_buffer import *
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


class Critic(tf.keras.Model):

    def __init__(self, name, hidden0=CRITIC_HIDDEN[0], hidden1=CRITIC_HIDDEN[1]):
        self.hidden0 = hidden0
        self.hidden1 = hidden1
        self.net_name = name
        self.dense0 = Dense(self.hidden0, activation='relu')
        self.dense1 = Dense(self.hidden1, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        state_action_value = self.dense0(tf.concat([state, action], axis=1))
        state_action_value = self.dense1(state_action_value)

        q_value = self.q_value(state_action_value)
        return q_value

class CriticValue(tf.keras.Model):
    def __init__(self, name, hidden0=CRITIC_HIDDEN[0], hidden1=CRITIC_HIDDEN[1]):
        super(CriticValue, self).__init__()
        self.hidden0 = hidden0
        self.hidden1 = hidden1
        self.net_name = name

        self.dense0 = Dense(self.hidden0, activation='relu')
        self.dense1 = Dense(self.hidden1, activation='relu')
        self.value = Dense(1, activation=None)

    def call(self, state):
        value = self.dense_0(state)
        value = self.dense_1(value)

        value = self.value(value)

        return value


#**** Actor Network ****
class Actor(tf.keras.Model):

    #**** Initialize Actor Network **** 
    # name: name of the network
    # upper_bound: upper bound of the action
    # actions_dim: dimension of the action
    # hidden0: first hidden layer size
    # hidden1: second hidden layer size
    # epsilon: epsilon Value
    # log_std_min: minimum log standard deviation
    # log_std_max: maximum log standard deviation
    def __init__(self, name, upper_bound, actions_dim,
                 hidden0=CRITIC_HIDDEN[0], hidden1=CRITIC_HIDDEN[1],
                 epsilon=EPSILON, log_std_min=LOG_STD_MIN,
                 log_std_max=LOG_STD_MAX):
        self.hidden0 = hidden0
        self.hidden1 = hidden1
        self.actions_dim = actions_dim
        self.net_name = name
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense0 = Dense(self.hidden0, activation='relu')
        self.dense1 = Dense(self.hidden1, activation='relu')
        self.mean = Dense(self.actions_dim, activation=None)
        self.log_std = Dense(self.actions_dim, activation=None)


    # **** Call Actor Network ****
    # state: state of the Environment
    # return: policy
    def call(self, state):
        policy = self.dense0(state) # pass current state through the first hidden layer
        policy = self.dense1(policy) # pass 
        mean = self.mean(policy)
        log_std = self.log_std(policy)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    # **** Get Action ****
    def get_action(self, state, reparameterize=True):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)

        z = tf.random.normal(shape=mean.shape, mean=0, stddev=1.)
        if reparameterize:
            actions = mean + std * z
        else:
            actions = normal_distr.sample() # change later to normal_distr.log_prob
        action = tf.math.tanh(actions) * self.upper_bound
        log_probs = normal_distr.log_prob(action) - tf.math.log(1- tf.math.pow(action, 2)
                                                                + self.epsilon)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs



