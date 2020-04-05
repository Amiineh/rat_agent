import json
import os


class Hyperparameters(object):

    def __init__(self,
                 learning_rate=0.0001,  # Q-network learning rate
                 nsteps=5,  # n-step updating
                 num_env=16,  # number of parallel agents (cpus)
                 gamestate=None,
                 reward_scale=1.0,
                 num_timesteps=6e7,
                 save_interval=125000,  # save
                 save_video_interval=0,  # Save video every x steps (0 = disabled)
                 save_video_length=1000,  # Length of recorded video
                 network='cnn',  # network type (mlp, cnn, lstm, cnn_lstm, conv_only)
                 seed=None,
                 ):

        self.learning_rate = learning_rate
        self.nsteps = nsteps
        self.num_env = num_env
        self.gamestate = gamestate
        self.reward_scale = reward_scale
        self.num_timesteps = num_timesteps
        self.save_interval = save_interval
        self.save_video_interval = save_video_interval
        self.save_video_length = save_video_length
        self.network = network
        self.seed = seed


class Environment(object):
    def __init__(self, name='DeepmindLabSoundTaskZero-v0', state_size=None, action_size=6, type='dmlab', fps=60):
        if state_size is None:
            state_size = [84, 84, 4]
        self.name = name
        self.state_size = state_size  # image size
        self.action_size = action_size
        self.type = type
        self.fps = fps


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

    for env_id in ['DeepmindLabNosePokeZero-v0', 'DeepmindLabNosePokeOne-v0', 'DeepmindLabNosePokeTwo-v0']:
        for network in ['cnn']:
            for lr in [0.1, 0.01, 0.001]:
                hyper = Hyperparameters(learning_rate=lr, network=network)
                exp = Experiment(id=idx_base, agent='acktr_dm', env_id=env_id, output_path='train_' + str(idx_base),
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
        json.dump(info, outfile, indent=4)


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
