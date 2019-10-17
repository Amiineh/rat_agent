import json
import os


class Hyperparameters(object):

    def __init__(self,
                 max_steps=5000,  # max steps in an episode
                 train_episodes=10000,  # max number of episodes
                 gamma=0.99,  # future reward discount
                 n=20,  # n-step updating
                 entropy_reg_term=1,  # regularization term for entropy
                 normalise_entropy=False,
                 # when true normalizes entropy to be in [-1, 0] to be more invariant to different size action spaces
                 explore_start=1.0,  # exploration probability at start
                 explore_stop=0.1,  # minimum exploration probability
                 decay_rate=0.002,  # exponential decay rate for exploration prob
                 explore_test=0.01,  # exploration rate for test time
                 kernel_size=None,
                 stride=None,
                 output_filters_conv=None,
                 hidden_size=256,  # number of units in each Q-network hidden layer
                 learning_rate=0.0001,  # Q-network learning rate
                 memory_size=1000000,  # memory capacity
                 batch_size=32,  # experience mini-batch size
                 num_envs=32,
                 save_log=100,  # save
                 ):

        if output_filters_conv is None:
            output_filters_conv = [16, 32]
        if kernel_size is None:
            kernel_size = [8, 4]
        if stride is None:
            stride = [4, 2]

        self.max_steps = max_steps
        self.train_episodes = train_episodes
        self.gamma = gamma
        self.n = n
        self.entropy_reg_term = entropy_reg_term
        self.normalise_entropy = normalise_entropy
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.explore_test = explore_test
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_filters_conv = output_filters_conv
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.save_log = save_log


class Environment(object):

    def __init__(self, name='Breakout-v0', state_size=None, action_size=4):
        if state_size is None:
            state_size = [84, 84, 4]
        self.name = name
        self.state_size = state_size  # image size
        self.action_size = action_size


class Experiment(object):

    def __init__(self, id, agent, env_id, output_path, train_completed=False, hyper=None, env=None):
        """
        :param id: index of output data folder
        :param agent: name of algorithm for agent (e.g. 'dqn', 'a3c')
        :param output_path: output directory
        """

        if hyper is None:
            hyper = Hyperparameters()
        if env is None:
            env = Environment(env_id)

        self.id = id
        self.agent = agent
        self.env_id = env_id
        self.output_path = output_path
        self.train_completed = train_completed
        self.hyper = hyper
        self.env = env


def decode_exp(dct):
    hyper = Hyperparameters()
    for key in hyper.__dict__.keys():
        if key in dct['hyper'].keys():
            hyper.__setattr__(key, dct['hyper'][key])
    env = Environment()
    for key in env.__dict__.keys():
        if key in dct['env'].keys():
            env.__setattr__(key, dct['env'][key])
    exp = Experiment(dct['id'], dct['agent'], dct['env_id'], dct['output_path'], dct['train_completed'], hyper, env)
    return exp


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
            hyper = Hyperparameters(learning_rate=lr)
            exp = Experiment(id=idx_base, agent='dqn_gym', env_id=env_id, output_path='train_' + str(idx_base),
                             hyper=hyper)

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

    print('retrieved experiment:')
    for key in exp.__dict__.keys():
        if key is 'hyper':
            print('hyper:', exp.hyper.__dict__)
        elif key is 'env':
            print('env:', exp.env.__dict__)
        else:
            print(key, ':', exp.__getattribute__(key))

    return exp

