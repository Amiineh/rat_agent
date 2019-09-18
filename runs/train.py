import agents
import os
import json


def run(opt, output_path, run):

    if opt.train_completed:
        print("Experiment already trained in " + opt.agent + "/" + opt.output_path)
        return

    id_path = output_path + opt.output_path
    dirname = os.path.dirname(id_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    run(opt, id_path)
    with open(output_path + 'train.json') as infile:
        info = json.load(infile)

    info[str(opt.id)]['train_completed'] = True

    with open(output_path + 'train.json', 'w') as outfile:
        json.dump(info, outfile)

