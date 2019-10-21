import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--agent', type=str, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
FLAGS = parser.parse_args()


output_path = {
    'amineh': '/Users/amineh.ahm/Desktop/Mice/code/rat_exp/',
    'om': '/om/user/amineh/rat_exp/',
    'om2': '/om2/user/amineh/rat_exp/'}[FLAGS.host_filesystem]

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
    from agents.dqn_gym import experiments
    from agents.dqn_gym.train import run
    output_path = output_path + "dqn_dm/"


def generate_experiments(id):
    # is is not used
    from runs import experiments as exp
    exp.run(output_path, experiments)


def run_train(id):
    from runs import train
    opt = experiments.get_experiment(output_path, id)
    train.run(opt, output_path, run)


def find_id(id):
    from runs import find_id as find
    opt = experiments.get_experiment(output_path, id)
    find.run(opt)


switcher = {
    'train': run_train,
    'find_id': find_id,
    'generate_experiments': generate_experiments,
}


switcher[FLAGS.run](FLAGS.experiment_index)
