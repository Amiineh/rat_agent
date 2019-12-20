from .vec_env import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, has_sound=False):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)
        self.has_sound = has_sound

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0

        if self.has_sound:
            self.stackedobs[..., -obs.shape[-1]-1:-1] = obs
            wos = self.venv.observation_space  # wrapped ob space
            white = np.ones([wos.shape[0], wos.shape[1]]) * 255
            grey = np.ones([wos.shape[0], wos.shape[1]]) * 128
            black = np.zeros([wos.shape[0], wos.shape[1]])
            for (i, info) in enumerate(infos):
                if info['sound_status']:
                    self.stackedobs[i, :, :, -1] = white
                elif info['distractor_status']:
                    self.stackedobs[i, :, :, -1] = grey
                else:
                    self.stackedobs[i, :, :, -1] = black
        else:
            self.stackedobs[..., -obs.shape[-1]:] = obs

        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
