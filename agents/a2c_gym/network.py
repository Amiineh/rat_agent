import tensorflow as tf
from agents.a2c_gym.baselines.common import tf_util
from agents.a2c_gym.baselines.utils import find_trainable_variables, Scheduler
from tensorflow import losses
import functools
import time
from agents.a2c_gym.baselines.common import set_global_seeds
from agents.a2c_gym.baselines.common.policies import build_policy
from agents.a2c_gym.runner import Runner
from collections import deque


# class ActorCriticNetwork:
#
#     def __init__(self, name, opt, env):
#         with tf.variable_scope(name):
#             self.name = name
#
#             self.inputs_ = tf.placeholder(tf.float32,
#                                           [None, opt.env.state_size[0], opt.env.state_size[1], opt.env.state_size[2]],
#                                           name='inputs')
#             self.conv1 = tf.contrib.layers.conv2d(self.inputs_, opt.hyper.output_filters_conv[0], kernel_size=opt.hyper.kernel_size[0], stride=opt.hyper.stride[0])
#             self.conv2 = tf.contrib.layers.conv2d(self.conv1, opt.hyper.output_filters_conv[1], kernel_size=opt.hyper.kernel_size[1], stride=opt.hyper.stride[1])
#
#             self.fc1 = tf.contrib.layers.fully_connected(
#                 tf.reshape(self.conv2, [-1, self.conv2.shape[1] * self.conv2.shape[2] * self.conv2.shape[3]]),
#                 opt.hyper.hidden_size)
#
#             # Value function - Linear output layer
#             self.value_output = tf.contrib.layers.fully_connected(self.fc1, 1,
#                                                                   activation_fn=None)
#
#             # Policy - softmax output layer
#             self.policy_logits = tf.contrib.layers.fully_connected(self.fc1, opt.env.action_size, activation_fn=None)
#             self.policy_output = tf.contrib.layers.softmax(self.policy_logits)
#
#             self.action_output = tf.squeeze(tf.multinomial(logits=self.policy_logits, num_samples=1), axis=1)
#
#             # baselines:
#             self.policy = build_policy(env, network, **network_kwargs)
#             self.sess = tf_util.get_session()
#             self.nbatch = opt.hyper.nenvs * opt.hyper.nsteps
#
#             with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
#                 # step_model is used for sampling
#                 self.step_model = self.policy(opt.hyper.nenvs, 1, self.sess)
#
#                 # train_model is used to train our network
#                 self.train_model = self.policy(self.nbatch, opt.hyper.nsteps, self.sess)
#
#             self.A = tf.placeholder(self.train_model.action.dtype, self.train_model.action.shape)
#             self.ADV = tf.placeholder(tf.float32, [self.nbatch])
#             self.R = tf.placeholder(tf.float32, [self.nbatch])
#             self.LR = tf.placeholder(tf.float32, [])
#
#             # Calculate the loss
#             # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
#
#             # Policy loss
#             self.neglogpac = self.train_model.pd.neglogp(self.A)
#             # L = A(s,a) * -logpi(a|s)
#             self.pg_loss = tf.reduce_mean(self.ADV * self.neglogpac)
#
#             # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
#             self.entropy = tf.reduce_mean(self.train_model.pd.entropy())
#
#             # Value loss
#             self.vf_loss = losses.mean_squared_error(tf.squeeze(self.train_model.vf), self.R)
#
#             self.loss = self.pg_loss - self.entropy * opt.hyper.ent_coef + self.vf_loss * opt.hyper.vf_coef
#
#             # Update parameters using loss
#             # 1. Get the model parameters
#             self.params = find_trainable_variables("a2c_model")
#
#             # 2. Calculate the gradients
#             self.rads = tf.gradients(self.loss, self.params)
#             if opt.hyper.max_grad_norm is not None:
#                 # Clip the gradients (normalize)
#                 self.grads, self.grad_norm = tf.clip_by_global_norm(self.grads, opt.hyper.max_grad_norm)
#             self.grads = list(zip(self.grads, self.params))
#             # zip aggregate each gradient with parameters associated
#             # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
#
#             # 3. Make op for one policy and value update step of A2C
#             self.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt.hyper.learning_rate, decay=opt.hyper.alpha, epsilon=opt.hyper.epsilon)
#
#             self.train_op = self.optimizer.apply_gradients(self.grads)
#
#             tf.summary.scalar('loss', self.loss)
#
#             self.reward_summary = tf.placeholder(tf.float32, name='reward_summary')
#             tf.summary.scalar('reward', self.reward_summary)


class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model
        train():
        - Make the training part (feedforward and retropropagation of gradients)
        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.
    Parameters:
    -----------
    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies
    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)
    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)
    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)
    total_timesteps:    int, total number of timesteps to train on (default: 80M)
    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)
    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)
    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)
    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)
    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output
    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)
    alpha:              float, RMSProp decay parameter (default: 0.99)
    gamma:              float, reward discounting parameter (default: 0.99)
    log_interval:       int, specifies how frequently the logs are printed out (default: 100)
    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''



    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
    return model















