import sys
sys.path.append("../src")
import gym # Import OpenAI gym
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

config = dict(
    learning_rate_actor = 0.0005,
    learning_rate_critic = 0.0005,
    batch_size = 64,
    arch = "SAC",
)

wandb.init(
  project=f"tensorflow2_sac_{ENV_NAME.lower()}",
  tags=["SAC", "FCL", "RL"],
  config=config,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v0")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--evaluate", metavar="path_to_model")
    args = parser.parse_args()

    if args.train:
        env = gym.make(args.env)
        agent = SACAgent(env, wandb.config)
        agent.train()
        agent.save()
