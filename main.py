import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import multiprocessing
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env
import sys
from baselines import logger
from runs.train import run
import importlib


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
    'vm': '/home/amineh/Shared/Mice/code/rat_exp/'}[FLAGS.host_filesystem]

output_path = output_path + FLAGS.agent + "/"
experiments = importlib.import_module("experiments." + FLAGS.agent)


def generate_experiments(id):
    # is is not used
    from runs import experiments as exp
    exp.run(output_path, experiments)


def run_train(id):
    opt = experiments.get_experiment(output_path, id)

    def configure_logger(log_path, **kwargs):
        if log_path is not None:
            logger.configure(log_path)
        else:
            logger.configure(**kwargs)

    def get_env_type(opt):
        env_type = opt.env.type
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
        env = make_vec_env(env_id, env_type=env_type, num_env=nenv, seed=seed, gamestate=opt.hyper.gamestate,
                           reward_scale=opt.hyper.reward_scale, opt=opt)
        has_sound = "sound" in opt.env.name.lower()  # add channel for sound
        frame_stack_size = frame_stack_size + 1 if has_sound else frame_stack_size
        env = VecFrameStack(env, frame_stack_size, has_sound)
        return env

    id_path = output_path + opt.output_path
    if not os.path.exists(id_path):
        os.makedirs(id_path)
    env_type, env_id = get_env_type(opt)
    print('env_type: {}'.format(env_type))
    env = build_env(opt)
    run(opt, output_path, env)


def find_id(id):
    from runs import find_id as find
    opt = experiments.get_experiment(output_path, id)
    find.run(opt)


def remove_id(id):
    from runs import remove_id as remove
    remove.run(id, output_path)


switcher = {
    'train': run_train,
    'find_id': find_id,
    'gen': generate_experiments,
    'remove': remove_id,
}


if __name__ == '__main__':
    switcher[FLAGS.run](FLAGS.experiment_index)


