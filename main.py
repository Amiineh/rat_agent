import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--agent', type=str, required=True)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
FLAGS = parser.parse_args()

code_path = {
    'amineh': '/Users/amineh.ahm/Desktop/Mice/code/rat_agent/',
    'om': '/om/user/amineh/rat_agent/'}[FLAGS.host_filesystem]

output_path = {
    'amineh': '/Users/amineh.ahm/Desktop/Mice/code/rat_exp/',
    'om': '/om/user/amineh/rat_exp/'}[FLAGS.host_filesystem]

if FLAGS.agent == "priint":
    from agents import priint as agent
    from agents.priint import experiments
    output_path = output_path + "priint/"

if FLAGS.agent == "random_gym":
    from agents.random_gym import experiments
    output_path = output_path + "random_gym/"


def generate_experiments(id):
    # is is not used
    from runs import experiments as exp
    exp.run(output_path, FLAGS.agent)


def run_train(id):
    from runs import train
    opt = experiments.get_experiment(output_path, id)
    train.run(opt, output_path)


def find_id(id):
    from runs import find_id as find
    return find.run(id)


switcher = {
    'train': run_train,
    'find_id': find_id,
    'generate_experiments': generate_experiments,
}


switcher[FLAGS.run](FLAGS.experiment_index)