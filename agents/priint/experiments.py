import copy
import json
import os


class Environment(object):

    def __init__(self, name='water_maze'):
        self.name = name


class DNN(object):

    def __init__(self, layers=2, name='priint'):
        self.name = name
        self.layers = layers


class Hyperparameters(object):

    def __init__(self, batch_size=32, learning_rate=0.001):
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class Experiment(object):

    def __init__(self, id, agent, output_path, train_completed=None, env=None, dnn=None, hyper=None):
        """
        :param id: index of output data folder
        :param agent: name of algorithm for agent (e.g. 'dqn', 'a3c')
        :param output_path: output directory
        """

        self.id = id
        self.agent = agent
        self.output_path = output_path
        if train_completed is None:
            self.train_completed = False
        else:
            self.train_completed = train_completed

        if env is None:
            self.env = Environment()
        else:
            self.env = env
        if dnn is None:
            self.dnn = DNN()
        else:
            self.dnn = dnn
        if hyper is None:
            self.hyper = Hyperparameters
        else:
            self.hyper = hyper
        

def decode_exp(dct):
    env = Environment(dct['env']['name'])
    dnn = DNN(dct['dnn']['layers'], dct['dnn']['name'])
    hyper = Hyperparameters(dct['hyper']['batch_size'], dct['hyper']['learning_rate'])
    return Experiment(dct['id'], dct['agent'], dct['output_path'], dct['train_completed'], env, dnn, hyper)


def exp_exists(exp, info):
    dict = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
    for idx in info:
        flag = True
        for key in info[idx]:
            if key == 'id' or key == 'output_path':
                continue
            if info[idx][key] != dict[key]:
                flag = False
        if flag:
            return idx
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
            hyper = Hyperparameters(batch_size, learning_rate)
            exp = Experiment(id=idx_base, agent='priint', output_path='train_'+str(idx_base), hyper=hyper)

            idx = exp_exists(exp, info)
            if idx is not False:
                print("exp already exists with id", idx)
                continue

            s = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
            print(s)
            info[str(idx_base)] = s
            idx_base += 1

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile)


def get_experiment(output_path, id):
    info_path = output_path + 'train.json'
    with open(info_path) as infile:
        trained = json.load(infile)
    opt = trained[str(id)]
    exp = decode_exp(opt)
    return exp

