import os
import json
import sys
import re
import multiprocessing
import os.path as osp
import gym
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

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(opt, env, id_path):
    env_type, env_id = get_env_type(opt)
    # print('env_type: {}'.format(env_type))

    total_timesteps = int(opt.hyper.num_timesteps)
    seed = opt.hyper.seed

    agent = opt.agent.split('_')[0]
    learn = get_learn_function(agent)
    alg_kwargs = get_learn_function_defaults(agent, env_type)

    # env = build_env(opt)
    # if opt.hyper.save_video_interval != 0:
    #     env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
    #                            record_video_trigger=lambda x: x % opt.hyper.save_video_interval == 0,
    #                            video_length=opt.hyper.save_video_length)

    if opt.hyper.network:
        alg_kwargs['network'] = opt.hyper.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(agent, env_type, env_id, alg_kwargs))

    load_path = None
    if osp.isfile(id_path+'/save.pkl'):
        load_path = id_path
    model = learn(
        env=env,
        seed=seed,
        nsteps=opt.hyper.nsteps,
        total_timesteps=total_timesteps,
        load_path=load_path,
        save_path=id_path,
        save_interval=opt.hyper.save_interval,
        lr=opt.hyper.learning_rate,
        **alg_kwargs
    )

    return model  # , env


def build_env(opt):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = opt.hyper.num_env or ncpu
    seed = opt.hyper.seed

    env_type, env_id = get_env_type(opt)

    frame_stack_size = 4
    env = make_vec_env(env_id, env_type='atari', num_env=nenv, seed=seed, gamestate=opt.hyper.gamestate,
                       reward_scale=opt.hyper.reward_scale)
    env = VecFrameStack(env, frame_stack_size)

    return env


def get_env_type(opt):
    env_type = 'atari'
    env_id = opt.env.name

    # # Re-parse the gym registry, since we could have new envs since last time.
    # for env in gym.envs.registry.all():
    #     env_type = env.entry_point.split(':')[0].split('.')[-1]
    #     _game_envs[env_type].add(env.id)  # This is a set so add is idempotent
    #
    # if env_id in _game_envs.keys():
    #     env_type = env_id
    #     env_id = [g for g in _game_envs[env_type]][0]
    # else:
    #     env_type = None
    #     for g, e in _game_envs.items():
    #         if env_id in e:
    #             env_type = g
    #             break
    #     if ':' in env_id:
    #         env_type = re.sub(r':.*', '', env_id)
    #     assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

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


def run(opt, output_path, env):
    if opt.train_completed:
        print("Experiment already trained in " + opt.agent + "/" + opt.output_path)
        return

    id_path = output_path + opt.output_path

    # configure logger, disable logging in child MPI processes (with rank > 0)

    # arg_parser = common_arg_parser()
    # args, unknown_args = arg_parser.parse_known_args(args)
    # extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(id_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(id_path, format_strs=[])

    model = train(opt, env, id_path)

    save_path = osp.expanduser(id_path)
    model.save(save_path)

    save_images(id_path, opt, model, env)

    if opt.hyper.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                # print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

    env.close()

    with open(output_path + 'train.json') as infile:
        info = json.load(infile)

    info[str(opt.id)]['train_completed'] = True

    with open(output_path + 'train.json', 'w') as outfile:
        json.dump(info, outfile)

    return model


