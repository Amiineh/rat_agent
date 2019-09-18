import json
import os


class Hyperparameters(object):

    def __init__(self):
        # General Parameters
        self.max_steps = 5000
        self.train_episodes = 500  # max number of episodes to learn from max_steps = 5000               # max steps in an episode
        self.gamma = 0.99  # future reward discount

        # Exploration parameters
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.002  # exponential decay rate for exploration prob

        # Network parameters
        self.kernel_size_1 = [8, 8, 3]
        self.output_filters_conv1 = 32
        self.output_filters_conv2 = 64
        self.output_filters_conv3 = 64
        self.hidden_size = 512  # number of units in each Q-network hidden layer
        self.learning_rate = 0.0001  # Q-network learning rate

        # Memory parameters
        self.memory_size = 10000  # memory capacity
        self.batch_size = 1  # experience mini-batch size
        self.pretrain_length = self.batch_size  # number experiences to pretrain the memory

        # target QN
        self.update_target_every = 2000


class Environment(object):

    def __init__(self, name='Breakout-v0'):
        self.name = name
        self.state_size = [80, 80, 3]
        self.action_size = 4


class Experiment(object):

    def __init__(self, id, agent, env_id, output_path, train_completed=None):
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

        self.hyper = Hyperparameters()
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

    for env_id in ['Breakout-v0']:
        exp = Experiment(id=idx_base, agent='dqn_gym', env_id=env_id, output_path='train_' + str(idx_base))

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

