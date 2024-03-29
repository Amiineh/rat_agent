import os
import gym
from gym import wrappers, logger
from agents.random_gym.network import RandomAgent


def run(opt, id_path):

    logger.set_level(logger.INFO)
    env = gym.make(opt.env_id)
    env = wrappers.Monitor(env, directory=id_path, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    env.close()