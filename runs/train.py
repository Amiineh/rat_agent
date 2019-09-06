import agents
import os
import json


def run(opt, output_path):
    if opt.agent == 'priint':
        from agents.priint.train import run
        global run

    if opt.train_completed:
        print("Experiment already trained in " + opt.agent + "/" + opt.output_path)
        return

    run(opt, output_path)
    with open(output_path + 'train.json') as infile:
        info = json.load(infile)

    info[str(opt.id)]['train_completed'] = True

    print(info)
    with open(output_path + 'train.json', 'w') as outfile:
        json.dump(info, outfile)

