import numpy as np
import random
import tensorflow as tf
import gym
from gym import wrappers, logger
from agents.a2c_gym.network import ActorCriticNetwork
from PIL import Image
import sys
import os
from os import listdir
from os.path import isfile, join
from multiprocessing import Process, Queue, Pipe
from agents.a2c_gym.network import learn



def get_last_ep(path):
    files = []
    if os.path.exists(path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
    if len(files) == 0:
        return 1
    last = files[-1].split('-')[1].split('.')[0]
    return int(last)


def save_images(id_path, opt, sess, targetQN, env, save_steps=100, num_repeats=4):
    action = get_random_action()
    done = None
    for step in range(save_steps):
        obs = [None for _ in range(num_repeats)]

        for rep in range(num_repeats):

            # todo: activate for deepmind_lab
            # if not env.is_running():
            #     env.reset()

            obs[rep], reward, done, info = env.step(action)
            img = Image.fromarray(obs[rep])
            img_path = os.path.join(id_path, 'images')
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            img.save(img_path + '/action_' + str(step) + '_' + str(rep) + '.jpg', 'JPEG')

            if done:
                break

        if done:
            env.reset()
            action = get_random_action()
        else:
            states = np.stack(obs, axis=2)
            action = eps_greedy(opt.hyper.explore_test, sess, targetQN, states)
    return


def get_action(mainA2C, sess, state, t_list, opt, print_policies=False):
    """
    Returns the action chosen by the QNetwork.
    Should be called by the mainA2C
    """
    feed = {mainA2C.inputs_: np.reshape(state, [-1, opt.hyper.state_size[0], opt.hyper.state_size[1], opt.hyper.state_size[2]])}
    actions, policies, logits = sess.run([mainA2C.action_output, mainA2C.policy_output, mainA2C.policy_logits], feed_dict=feed)
    actions_softmax = [np.random.choice(len(policy), p=policy) for policy in policies]
    print("actions action op: ", actions)
    print("actions softmax: ", actions_softmax)
    print("logits: ", logits)
    return actions


def get_value(mainA2C, sess, state, opt):
    """
    Returns the value of a state by the QNetwork.
    Should be called by the mainA2C
    """
    feed = {mainA2C.inputs_: np.reshape(state, [-1, opt.hyper.state_size[0], opt.hyper.state_size[1], opt.hyper.state_size[2]])}
    policies, values = sess.run([mainA2C.policy_output, mainA2C.value_output], feed_dict=feed)
    return values


def train_step(mainA2C, sess, states, actions, rewards, discounts, initial_Rs):
    """
    Runs a train step
    Returns the loss
    Should be called by mainA2C
    """
    loss, extra, opt = sess.run([mainA2C.loss, mainA2C.extra, mainA2C.train_op],
                                feed_dict={mainA2C.inputs_: np.reshape(states, [-1, 80, 80, 3]),
                                           mainA2C.actions_: actions,
                                           mainA2C.rewards: rewards,
                                           mainA2C.discounts: discounts,
                                           mainA2C.initial_Rs: initial_Rs})
    return loss, extra


def get_random_action(action_size=4):
    return random.randint(0, action_size - 1)


def index_to_english(env, action):
    # for Breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    english_names_of_actions = env.unwrapped.get_action_meanings()
    return english_names_of_actions[action]


def env_worker(child_conn, opt):
    # todo: deepmind_lab
    # def env_worker(child_conn, level, config):
    # env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    # env.reset()
    logger.set_level(logger.INFO)
    env = gym.make(opt.env_id)
    env = wrappers.AtariPreprocessing(env)
    env._max_episode_steps = opt.hyper.max_steps
    # env = wrappers.Monitor(env, directory=id_path + '/video', force=True)
    env.reset()
    print("Started another environment worker!")

    while True:
        # if child_conn.poll():
        # Note: using the above loops, without it blocks. Not sure which is fastest.
            action, t = child_conn.recv()
            package = env_step(env, action, t)
            child_conn.send(package)


def env_step(env, action, num_repeats=4):
    R = 0.0
    obs = [None for _ in range(num_repeats)]

    for t in range(num_repeats):

        # todo: activate for deepmind_lab
        # if not env.is_running():
        #     env.reset()

        obs[t], reward, done, info = env.step(action)
        R += reward

        if done:
            env.reset()

    observations = np.stack(obs, axis=2)
    next_state = observations
    # todo: add for deepmind_lab
    # next_state = env.observations()['RGB_INTERLEAVED']
    # # next_state = np.reshape(next_state, [-1])

    return next_state, R, done


def env_step(env, action, t, num_repeats=20):
    # print(index_to_english(action))
    english_action = index_to_english(action)
    action = map_to_dmlab(action)
    reward = 0
    count = 0
    reset = False

    while count < num_repeats:

        if not env.is_running():
            env.reset()

        reward = env.step(action)

        if reward != 0:
            break

        count += 1
    if reward > 0:
        print("Action: ", english_action, " REWARD: " + str(reward), "Steps taken: ", t)

    # Dealing w/ end of episode
    next_state = None
    episode_done = False
    # print("t: ", t, "max_steps: ", max_steps, "reward: ", reward)
    if reward > 0 or t == max_steps:
        if t == max_steps:
            reward = .0000001
            # reward = -1
        next_state = np.zeros(state_size)
        t = 0
        env.reset()
        next_state = env.observations()['RGB_INTERLEAVED']
        episode_done = True

    else:
        next_state = env.observations()['RGB_INTERLEAVED']
        t += 1

    return (next_state, reward, t, episode_done)


def get_bootstrap(args, sess, mainA2C):
    # Getting R to use as initial condition. Essentially doing the whole target function thing.
    state, action, next_state, reward, t, episode_done = args

    if reward == 0:
        next_state_data = np.expand_dims(np.array(next_state), axis=0)
        bootstrapped_R = np.max(mainA2C.get_value(sess, next_state_data))  # Shouldnt need to be a max
    else:
        bootstrapped_R = 0

    return bootstrapped_R


def deep_cast_to_nparray(bad_array):
    return np.array([np.array([np.array(a) for a in inner]) for inner in bad_array])


def get_discounts(reward_list, opt):
    f = lambda x: 0.0 if x != 0 else opt.hyper.gamma
    return np.array([[f(x) for x in y] for y in reward_list])


def train(opt, mainA2C, id_path):
    print('started training...')
    sys.stdout.flush()
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)

    # Initialization
    envs_list = [deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)] * num_envs
    envs_list = map(reset_envs, envs_list)
    state_batch = map(lambda env: env.observations()['RGB_INTERLEAVED'], envs_list)
    next_state_batch = copy.deepcopy(state_batch)

    # Initalization of multiprocessing stuff
    pipes = [Pipe() for i in range(opt.hyper.num_envs)]
    parent_conns, child_conns = zip(*pipes)

    processes = [Process(target=env_worker,
          args=(child_conns[i], opt))
          for i in range(opt.hyper.num_envs)]

    for i in range(opt.hyper.num_envs):
        processes[i].start()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(id_path, sess.graph)
        if os.path.exists(id_path + '/saved/'):
            saver.restore(sess, tf.train.latest_checkpoint(id_path + '/saved/'))
            print('Restored model.')
        last_ep = get_last_ep(id_path + '/saved/')
        print('Running model from episode: {}'.format(last_ep))

        step = 0
        t_list = [0 for i in range(opt.hyper.num_envs)]
        for ep in range(last_ep, opt.hyper.train_episodes):
            # n-steps
            n_steps_parallel = []
            for i in range(opt.hyper.n):
                state_batch = next_state_batch  # TODO: Namespace collision?

                step += 1
                print("step: ", step)
                print_policies = True
                # if step % 10 == 0:
                #     print_policies = True
                # GPU, PARALLEL
                action_list = get_action(mainA2C, np.array(state_batch), t_list, print_policies)
                # action_list = map(apply_epsilon_greedy, zip(action_list, [step] * num_envs))

                # CPU, PARALLEL
                # Take action in environment, get new state and reward
                for i in range(opt.hyper.num_envs):
                    package = (action_list[i], t_list[i])
                    parent_conns[i].send(package)

                nextstate_reward_t_episodedone_list = [parent_conns[i].recv() for i in range(opt.hyper.num_envs)]
                next_state_batch, reward_list, t_list, episode_done_list = zip(*nextstate_reward_t_episodedone_list)

                env_tuples = zip(state_batch, action_list, next_state_batch, reward_list, t_list, episode_done_list)

                # Accumulate n-step experience
                n_steps_parallel.append(np.array(env_tuples))

            bootstrap_vals_list = [get_bootstrap(last_state, sess, mainA2C) for last_state in n_steps_parallel[0]]

            n_steps_parallel = [deep_cast_to_nparray(tup) for tup in np.moveaxis(n_steps_parallel, -1, 0)]
            state_list, action_list, _next_state_list, reward_list, _t_list, _episode_done_list = n_steps_parallel

            pcontinues_list = get_discounts(reward_list)

            print("action_list.shape: ", action_list)
            print("state_list.shape: ", state_list.shape)
            print("reward_list: ", reward_list)
            print("bootstrap_vals_list: ", bootstrap_vals_list)
            print("pcontinues_list: ", pcontinues_list)

            # Train step
            loss, extra = train_step(sess, state_list, action_list, reward_list, pcontinues_list,
                                             bootstrap_vals_list)

            print("total loss: ", loss)
            print("entropy: ", extra.entropy)
            print("entropy_loss: ", extra.entropy_loss)
            print("baseline_loss: ", extra.baseline_loss)
            print("policy_gradient_loss: ", extra.policy_gradient_loss)
            print("advantages: ", np.reshape(extra.advantages, [n, num_envs]))
            print("discounted_returns: ", np.reshape(extra.discounted_returns, [n, num_envs]))

            # TODO:
            # 1) Clip rewards
            # 2) Make min/max policy?
            # 3) clip policy gradients? Might already be done
            # 4) add summary

            # print("Saving...")
            # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

            # print("Resoring...")
            # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))

            # train_writer.add_summary(summ, ep)

            if ep % opt.hyper.save_log == 0:
                print("\nSaving graph...")
                saver.save(sess, id_path + '/saved/ep', global_step=ep, write_meta_graph=False)
                print("\nSaving images...")
                sys.stdout.flush()
                # save_images(id_path, opt, sess, targetQN, env)


def run(opt, id_path):
    tf.reset_default_graph()
    model = learn(
        env=opt.env.name,
        total_timesteps=opt.hyper.total_timesteps,
        # todo: possible args
        #  **alg_kwargs
    )


    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew[0] if isinstance(env, VecEnv) else rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()












