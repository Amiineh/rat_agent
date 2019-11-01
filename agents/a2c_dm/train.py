import sys
import re
import multiprocessing
import os.path as osp
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from PIL import Image
import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None


def train(opt, env, id_path):
    env_type, env_id = get_env_type(opt)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(opt.hyper.num_timesteps)
    seed = opt.hyper.seed

    learn = get_learn_function('a2c')
    alg_kwargs = get_learn_function_defaults('a2c', env_type)

    if opt.hyper.network:
        alg_kwargs['network'] = opt.hyper.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format('a2c', env_type, env_id, alg_kwargs))

    load_path = None
    if osp.isfile(id_path+'/save.pkl'):
        load_path = id_path
    model = learn(
        env=env,
        seed=seed,
        nsteps=opt.hyper.nsteps,
        lr=opt.hyper.learning_rate,
        save_interval=opt.hyper.save_interval,
        total_timesteps=total_timesteps,
        load_path=load_path,
        save_path=id_path,
        **alg_kwargs
    )

    return model


def build_env(opt):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = opt.hyper.num_env or ncpu
    seed = opt.hyper.seed

    env_type, env_id = get_env_type(opt)

    frame_stack_size = 4
    env = make_vec_env(env_id, env_type=env_type, num_env=nenv, seed=seed, gamestate=opt.hyper.gamestate,
                       reward_scale=opt.hyper.reward_scale)
    env = VecFrameStack(env, frame_stack_size)

    return env


def get_env_type(opt):
    env_type = opt.env.type
    env_id = opt.env.name
    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def save_images(id_path, opt, model, env):
    logger.log("Saving images")
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    episode = 0
    for step in range(opt.hyper.save_video_length):
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew

        for rep in range(4):
            img = Image.fromarray(obs[0, :, :, rep])
            img_path = os.path.join(id_path, 'images')
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            img.save(img_path + '/ep_' + str(episode) + '_step_' + str(step) + '.jpg', 'JPEG')

        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode {}, episode_rew={}'.format(episode, episode_rew))
            episode_rew = 0
            episode += 1
            obs = env.reset()


def run(opt, id_path, env):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        configure_logger(id_path)
    else:
        configure_logger(id_path, format_strs=[])

    model = train(opt, env, id_path)

    save_path = osp.expanduser(id_path)
    model.save(save_path)

    save_images(id_path, opt, model, env)

    env.close()

    return model


