import json
import os


class Hyperparameters(object):

    def __init__(self, lr=0.0001):
        # General Parameters
        self.max_steps = 5000   # max steps in an episode
        self.train_episodes = 10000  # max number of episodes
        self.gamma = 0.99  # future reward discount

        # Exploration parameters
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.1  # minimum exploration probability
        self.decay_rate = 0.002  # exponential decay rate for exploration prob
        self.explore_test = 0.01  # exploration rate for test time

        # Network parameters
        self.kernel_size = [8, 4, 3]
        self.stride = [4, 2, 1]
        self.output_filters_conv = [32, 64, 64]
        self.hidden_size = 512  # number of units in each Q-network hidden layer
        self.learning_rate = lr  # Q-network learning rate

        # Memory parameters
        self.memory_size = 1000000  # memory capacity
        self.batch_size = 32  # experience mini-batch size
        self.pretrain_length = self.memory_size  # number experiences to pretrain the memory

        # target QN
        self.update_target_every = 2001

        # save
        self.save_log = 100


class Environment(object):

    def __init__(self, name='Breakout-v0'):
        self.name = name
        self.state_size = [84, 84, 4]  # image size
        self.action_size = 4


class Experiment(object):

    def __init__(self, id, agent, env_id, output_path, train_completed=None, hyper=None):
        """
        :param id: index of output data folder
        :param agent: name of algorithm for agent (e.g. 'dqn', 'a3c')
        :param output_path: output directory
        """

        self.id = id
        self.agent = agent
        self.env_id = env_id
        self.output_path = output_path
        if train_completed is None:
            self.train_completed = False
        else:
            self.train_completed = train_completed

        if hyper is None:
            self.hyper = Hyperparameters()
        else:
            self.hyper = hyper
        self.env = Environment(env_id)


def decode_exp(dct):
    return Experiment(dct['id'], dct['agent'], dct['env_id'], dct['output_path'], dct['train_completed'])


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

    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        for env_id in ['Breakout-v0']:
            hyper = Hyperparameters(lr=lr)
            exp = Experiment(id=idx_base, agent='dqn_gym', env_id=env_id, output_path='train_' + str(idx_base), hyper=hyper)

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
