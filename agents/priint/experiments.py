import copy
import json
import os


class Environment(object):

    def __init__(self, name='water_maze'):
        self.name = name


class DNN(object):

    def __init__(self):
        self.name = 'priint'
        self.pretrained = False
        self.version = 1
        self.layers = 0


class Hyperparameters(object):

    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 1e-2
        self.drop_train = 1
        self.drop_test = 0
        self.momentum = 0.9


class Experiment(object):

    def __init__(self, id, agent, level, output_path):
        """
        :param id: index of output data folder
        :param agent: name of algorithm for agent (e.g. 'dqn', 'a3c')
        :param level: name of the level_script of the environment (e.g. 'morris_water_maze')
        :param output_path: output directory
        """

        self.id = id
        self.agent = agent
        # todo: add Environment instead of level
        self.level = level
        self.output_path = output_path
        self.train_completed = False

        self.dnn = DNN()
        self.hyper = Hyperparameters()


def exp_exists(opt, output_path):
    info_path = output_path + 'train.json'
    if os.path.isfile(info_path):
        with open(info_path, 'r') as infile:
            trained = json.load(infile)
            for key in trained:
                # todo: find a fancy way to do this
                if trained[key]['agent'] == opt.agent and \
                        trained[key]['level'] == opt.level and \
                        trained[key]['dnn']['name'] == opt.dnn.name and \
                        trained[key]['dnn']['pretrained'] == opt.dnn.pretrained and \
                        trained[key]['dnn']['version'] == opt.dnn.version and \
                        trained[key]['dnn']['layers'] == opt.dnn.layers and \
                        trained[key]['hyper']['batch_size'] == opt.hyper.batch_size and \
                        trained[key]['hyper']['learning_rate'] == opt.hyper.learning_rate and \
                        trained[key]['hyper']['drop_train'] == opt.hyper.drop_train and \
                        trained[key]['hyper']['drop_test'] == opt.hyper.drop_test and \
                        trained[key]['hyper']['momentum'] == opt.hyper.momentum :
                    infile.close()
                    return True
    return False


def generate_experiments(output_path):
    info = {}

    info_path = output_path + 'train.json'
    dirname = os.path.dirname(info_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0
    elif os.path.isfile(info_path):
        with open(info_path) as infile:
            info = json.load(infile)
            idx_base = int(list(info.keys())[-1]) + 1
    else:
        idx_base = 0

    for learning_rate in [0.01, 0.001]:
        for batch_size in [32, 64, 128]:
            opt = Experiment(id=idx_base, agent='priint', level='eg_water_maze', output_path=output_path)
            opt.hyper.learning_rate = learning_rate
            opt.hyper.batch_size = batch_size

            if exp_exists(opt, output_path):
                print("Experiment already exists, skiping...")
                continue

            info[idx_base] = {
                'id': idx_base,
                'agent': opt.agent,
                'level': opt.level,
                'output_path': 'train_' + str(opt.id),

                'dnn':{
                    'name': opt.dnn.name,
                    'pretrained': opt.dnn.pretrained,
                    'version': opt.dnn.version,
                    'layers': opt.dnn.layers,
                },

                'hyper':{
                    'batch_size': opt.hyper.batch_size,
                    'learning_rate': opt.hyper.learning_rate,
                    'drop_train': opt.hyper.drop_train,
                    'drop_test': opt.hyper.drop_test,
                    'momentum': opt.hyper.momentum,
                },
                'train_completed': False,
            }

            idx_base += 1

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile)


def get_experiment(output_path, id):
    info_path = output_path + 'train.json'
    with open(info_path) as infile:
        trained = json.load(infile)
    opt = trained[str(id)]

    # todo: find a fancy way to do this
    exp = Experiment(opt['id'], opt['agent'], opt['level'], opt['output_path'])
    exp.dnn.name = opt['dnn']['layers']
    exp.dnn.pretrained = opt['dnn']['pretrained']
    exp.dnn.version = opt['dnn']['version']
    exp.dnn.layers = opt['dnn']['layers']
    exp.hyper.batch_size = opt['hyper']['batch_size']
    exp.hyper.learning_rate = opt['hyper']['learning_rate']
    exp.hyper.drop_train = opt['hyper']['drop_train']
    exp.hyper.drop_test = opt['hyper']['drop_test']
    exp.hyper.momentum = opt['hyper']['momentum']
    exp.train_completed = opt['train_completed']

    return exp

