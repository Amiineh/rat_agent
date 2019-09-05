import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse


parser = argparse.ArgumentParser()
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


def run_train(opt):
    from runs import train
    train.run(opt)


switcher = {
    'train': run_train,
}


opts = experiments.get_experiments(output_path)
for opt in opts:
    switcher[FLAGS.run](opt)