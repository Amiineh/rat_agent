import agents
import os

def run(opt):
    if os.path.isfile(opt.output_path + str(opt.id) + 'info.json'):
        print("Json file already exists.")
        quit()

    if opt.dnn.name == 'priint':
        from agents.priint import train

    train.run(opt)