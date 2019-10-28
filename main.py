import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import multiprocessing
import os.path as osp
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
import sys
from baselines import logger


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--agent', type=str, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
FLAGS = parser.parse_args()


output_path = {
    'amineh': '/Users/amineh.ahm/Desktop/Mice/code/rat_exp/',
    'om': '/om/user/amineh/rat_exp/',
    'om2': '/om2/user/amineh/rat_exp/',
    'vm': '/home/amineh/Shared/Desktop/Mice/code/rat_exp/'}[FLAGS.host_filesystem]

if FLAGS.agent == "priint":
    from agents.priint import experiments
    from agents.priint.train import run
    output_path = output_path + "priint/"

if FLAGS.agent == "random_gym":
    from agents.random_gym import experiments
    from agents.random_gym.train import run
    output_path = output_path + "random_gym/"

if FLAGS.agent == "dqn_gym":
    from agents.dqn_gym import experiments
    from agents.dqn_gym.train import run
    output_path = output_path + "dqn_gym/"

if FLAGS.agent == "a2c_gym":
    from agents.a2c_gym import experiments
    from agents.a2c_gym.train import run
    output_path = output_path + "a2c_gym/"


if FLAGS.agent == "dqn_dm":
    from agents.dqn_dm import experiments
    from agents.dqn_dm.train import run
    output_path = output_path + "dqn_dm/"


def generate_experiments(id):
    # is is not used
    from runs import experiments as exp
    exp.run(output_path, experiments)


def run_train(id):
    from runs import train
    opt = experiments.get_experiment(output_path, id)

    def configure_logger(log_path, **kwargs):
        if log_path is not None:
            logger.configure(log_path)
        else:
            logger.configure(**kwargs)

    def get_env_type(opt):
        env_type = 'atari'
        env_id = opt.env.name
        return env_type, env_id

    def build_env(opt):
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin': ncpu //= 2
        nenv = opt.hyper.num_env or ncpu
        seed = opt.hyper.seed
        print("num environments: ", nenv)

        env_type, env_id = get_env_type(opt)

        frame_stack_size = 4
        configure_logger(id_path)
        env = make_vec_env(env_id, env_type='atari', num_env=nenv, seed=seed, gamestate=opt.hyper.gamestate,
                           reward_scale=opt.hyper.reward_scale)
        env = VecFrameStack(env, frame_stack_size)
        return env

    if opt.agent == "a2c_gym":
        id_path = output_path + opt.output_path

        env_type, env_id = get_env_type(opt)
        print('env_type: {}'.format(env_type))
        env = build_env(opt)
        if opt.hyper.save_video_interval != 0:
            env = VecVideoRecorder(env, osp.join(id_path, "videos"),
                                   record_video_trigger=lambda x: x % opt.hyper.save_video_interval == 0,
                                   video_length=opt.hyper.save_video_length)
        from agents.a2c_gym.train import run as a2c_run
        a2c_run(opt, id_path, env)
    else:
        train.run(opt, output_path, run)


def find_id(id):
    from runs import find_id as find
    opt = experiments.get_experiment(output_path, id)
    find.run(opt)


switcher = {
    'train': run_train,
    'find_id': find_id,
    'gen': generate_experiments,
}


if __name__ == '__main__':
    switcher[FLAGS.run](FLAGS.experiment_index)


