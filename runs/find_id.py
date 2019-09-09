import json


learning_rate = 0.01
batch_size = 64

def run(opt):
    with open(opt[0].output_path + 'train.json', 'r') as info_file:
        info = json.load(info_file)
        for id in info:
            if info[id]['hyper']['learning_rate'] == learning_rate and \
                info[id]['hyper']['batch_size'] == batch_size:
                print(id)


