import sys
sys.path.append("../src")
import gymnasium as gym # Import OpenAI gym
# import pybullet_envs # py
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import wandb # potentialy useful for logging
from config import *
from replay_buffer import *
from networks import *
from agent import *
import gymnasium as gym
import time
import argparse

from stable_baselines3 import SAC

# env = gym.make("Walker2d-v4", render_mode="rgb_array")
#
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=2000)
#
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     if done:
#         obs = vec_env.reset()

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v4")
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-e", "--evaluate", metavar="path_to_model")
args = parser.parse_args()

print(ACTOR_LR, CRITIC_LR, BATCH_SIZE)
config = dict(
    learning_rate_actor=ACTOR_LR,
    learning_rate_critic=CRITIC_LR,
    batch_size=BATCH_SIZE,
    arch="SAC",
    env=args.env,
)

wandb.init(
  project=f"tensorflow2_sac_{args.env.lower()}",
  tags=["SAC", "FCL", "RL"],
  config=config,
)

env = gym.make(args.env)
# replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE)
agent = Agent(env)

if __name__ == "__main__":

    scores = []
    evaluation = True

    if args.train:

        for _ in tqdm(range(MAX_GAMES)):
            # start_time = time.time()
            # env = gym.make(args.env)
            # agent = Agent(env)
            # env = MonitorStateWrapper(env)  # Wrap the environment to record state transitions
            # states = env.reset()
            # done = False
            # score = 0
            #
            # while not done:
            #     action = agent.get_action(states)
            #     new_states, reward, done, info = env.step(action)
            #     score += reward
            #
            #     agent.replay_buffer.append((states, action, reward, new_states, done)) 
            #     agent.learn() 
            #     self.replay_buffer.add(state[0], action[0], reward[0], next_statei[0], done[0])
            #     states = new_states
            #
            # scores.append(score)
            #
            # wandb.log({'Game number': _, 'Average reward': round(np.mean(scores[-10:]), 2),
            #            "Time taken": round(time.time() - start_time, 2)})
            #
            # if (_ + 1) % SAVE_FREQUENCY == 0:
            #     print("saving...")
            #     agent.save()
            #     print("saved")
                # for _ in tqdm(range(MAX_GAMES)):
            start_time = time.time()
            states = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.get_action(states)
                new_states, reward, done, info = env.step(action)
                score += reward
                agent.add_to_replay_buffer(states, action,
                                           reward, new_states, done)
                agent.learn()
                states = new_states
            scores.append(score)
            agent.replay_buffer.update_n_games()

            wandb.log({'Game number': agent.replay_buffer.n_games,
                       '# Episodes': agent.replay_buffer.buffer_counter,
                       "Average reward": round(np.mean(scores[-10:]), 2), \
                       "Time taken": round(time.time() - start_time, 2)})
    else:
        PATH_LOAD = args.evaluate
        observation = env.reset()
        action, log_probs = agent.actor.get_action_log_probs(observation[None, :], False)
        agent.actor(observation[None, :])
        agent.critic_0(observation[None, :], action)
        agent.critic_1(observation[None, :], action)
        agent.critic_value(observation[None, :])
        agent.critic_target_value(observation[None, :])
        agent.load()
    #     env = gym.make(args.env)
    #     agent = Agent(env)
    #     for _ int tdqm(range(MAX_EPISODES)):
    #         total_reward = 0
    #         while not done:
    #             action = agent.get_action(state)
    #             next_state, reward, done, info = env.step(action)
    #
    #             agent.replay_buffer.add(state, action, reward, next_state, done)
    #             state = next_state
    #             total_reward += reward
    #         agent.learn()
    #         wandb.log({"total_reward": total_reward})
