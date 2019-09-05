import copy
import json
import os


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
        self.level = level
        self.output_path = output_path

        self.dnn = DNN()
        self.hyper = Hyperparameters()


def get_experiments(output_path):
    info = {}
    opt = []

    idx_base = 0

    for learning_rate in [0.01, 0.001]:
        for batch_size in [32, 64, 128]:
            opt_handle = Experiment(id=idx_base, agent='priint', level='eg_water_maze', output_path=output_path)
            opt_handle.hyper.learning_rate = learning_rate
            opt_handle.hyper.batch_size = batch_size
            opt += [copy.deepcopy(opt_handle)]

            info[idx_base] = {
                'agent': opt_handle.agent,
                'level': opt_handle.level,
                'output_path': opt_handle.output_path + str(opt_handle.id),

                'dnn':{
                    'name': opt_handle.dnn.name,
                    'pretrained': opt_handle.dnn.pretrained,
                    'version': opt_handle.dnn.version,
                    'layers': opt_handle.dnn.layers,
                },

                'hyper':{
                    'batch_size': opt_handle.hyper.batch_size,
                    'learning_rate': opt_handle.hyper.learning_rate,
                    'drop_train': opt_handle.hyper.drop_train,
                    'drop_test': opt_handle.hyper.drop_test,
                    'momentum': opt_handle.hyper.momentum,
                }
            }

            idx_base += 1

    info_path = output_path + 'info.json'
    dirname = os.path.dirname(info_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile)

    return opt

