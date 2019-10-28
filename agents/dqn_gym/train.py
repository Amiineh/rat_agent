import numpy as np
import random
import tensorflow as tf
import gym
from gym import wrappers, logger
from agents.dqn_gym.memory import Memory
from agents.dqn_gym.network import QNetwork
from PIL import Image
import sys
import os
from os import listdir
from os.path import isfile, join
import gc

memory = Memory()


def get_last_step(path):
    files = []
    if os.path.exists(path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
    if len(files) == 0:
        return 0, 0
    ep, step = files[-1].split('.')[0].split('_')
    ep = ep.split('-')[1]
    step = step.split('-')[1]
    return int(ep), int(step)


def save_images(id_path, opt, sess, targetQN, env, save_steps=100, num_repeats=4):
    action = get_random_action()
    done = None
    for step in range(save_steps):
        obs = [None for _ in range(num_repeats)]

        for rep in range(num_repeats):
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


def update_target_variables(mainQN, targetQN, tau=1.0):
    q_vars = mainQN.get_network_variables()
    q_target_vars = targetQN.get_network_variables()
    update_target_op = [v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q_target_vars, q_vars)]
    return update_target_op


def get_epsilon(step, opt):
    # eps = 0.1 + 0.9 * (1M - step) / 1M
    eps = opt.hyper.explore_stop + \
          max(0, (opt.hyper.explore_start - opt.hyper.explore_stop) *
              ((opt.hyper.explore_duration - step) / opt.hyper.explore_duration))
    return eps


def eps_greedy(explore_p, sess, model, state):
    if explore_p > np.random.rand():
        # Make a random action
        action = get_random_action()
    else:
        # Get action from Q-network
        feed = {model.inputs_: state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))}
        Qs = sess.run(model.output, feed_dict=feed)
        action = np.argmax(Qs)
    return action


def get_random_action():
    return random.randint(0, 3)


def clip(reward):
    return np.clip(reward, -1, 1)


def index_to_english(env, action):
    # for Breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    english_names_of_actions = env.unwrapped.get_action_meanings()
    return english_names_of_actions[action]


def env_step(env, action, num_repeats=4):
    R = 0.0
    obs = [None for _ in range(num_repeats)]

    for t in range(num_repeats):
        obs[t], reward, done, info = env.step(action)
        R += reward

        if done:
            env.reset()

    observations = np.stack(obs, axis=2)
    next_state = observations
    return next_state, R, done


def pretrain(env, opt):
    print('pretraining...')
    sys.stdout.flush()
    state, reward, done = env_step(env, get_random_action())

    # Make a bunch of random actions and store the experiences
    for _ in range(opt.hyper.pretrain_length):

        action = get_random_action()
        next_state, reward, done = env_step(env, action)
        reward = clip(reward)
        memory.add((np.uint8(state), action, reward, np.uint8(next_state), done))

        if done:
            # Start new episode
            env.reset()
        else:
            state = next_state

        print('memory len: {}'.format(len(memory.buffer)))
        sys.stdout.flush()

    print('pretraining is done.')
    print('memory length: ', len(memory.buffer))
    sys.stdout.flush()
    return state


def train(env, state, opt, mainQN, targetQN, update_target_op, id_path):
    global summ
    print('started training...')
    sys.stdout.flush()
    saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1)

    gc.collect()
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Saver
        if os.path.exists(id_path + '/saved/'):
            saver.restore(sess, tf.train.latest_checkpoint(id_path + '/saved/'))
            print('Restored model.')
        last_episode, last_step = get_last_step(id_path + '/saved/')
        print('Running model from episode {} and step: {}'.format(last_episode, last_step))

        # Summaries
        train_writer = tf.summary.FileWriter(id_path, sess.graph)
        merge = tf.summary.merge_all()

        total_reward = 0
        max_reward = 0
        episode_reward = 0
        num_episodes = last_episode
        for step in range(last_step, opt.hyper.max_steps):

            # update target q network
            if step % opt.hyper.update_target_every == 0:
                sess.run(update_target_op)
                print("\nCopied model parameters to target network.")
                sys.stdout.flush()

            # Explore or Exploit
            explore_p = get_epsilon(step, opt)
            action = eps_greedy(explore_p, sess, mainQN, state)

            # Take action, get new state and reward
            next_state, reward, done = env_step(env, action)
            episode_reward += reward
            reward = clip(reward)
            memory.add((np.uint8(state), action, reward, np.uint8(next_state), done))

            if done:
                # Reset environment
                env.reset()

                total_reward = episode_reward
                num_episodes += 1
                max_reward = max(max_reward, episode_reward)

                print('\nEpisode: {}'.format(num_episodes),
                      '\nStep: {}'.format(step),
                      '\nEpisode reward: {}'.format(episode_reward),
                      '\nMax reward: {}'.format(max_reward),
                      '\nExplore P: {:.4f}'.format(explore_p),
                      '\nmemory len: {}'.format(len(memory.buffer)))
                sys.stdout.flush()
                episode_reward = 0

            else:
                state = next_state

            if step % opt.hyper.train_freq == 0:
                # Sample mini-batch from memory
                batch = memory.sample(opt.hyper.batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                dones = np.array([each[4] for each in batch])

                target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})
                # Set target_Qs to 0 for states where episode ends
                episode_ends = dones.all()
                target_Qs[episode_ends] = 0

                loss, _, summ = sess.run([mainQN.loss, mainQN.train_op, merge],
                                         feed_dict={mainQN.inputs_: states,
                                                    mainQN.targetQs_: target_Qs,
                                                    mainQN.reward: rewards,
                                                    mainQN.action: actions,
                                                    mainQN.reward_summary: total_reward})

                train_writer.add_summary(summ, step)

            if step % 1000 == 0:
                gc.collect()

            if step > last_step and step % opt.hyper.save_freq == 0:
                print("\nSaving graph...")
                saver.save(sess, id_path + '/saved/ep-' + str(num_episodes) + '_step', global_step=step,
                           write_meta_graph=False)
                print("\nSaving images...")
                sys.stdout.flush()
                save_images(id_path, opt, sess, targetQN, env)


def run(opt, id_path):
    tf.reset_default_graph()
    memory.set_capacity(opt.hyper.memory_size)
    mainQN = QNetwork(name='main_qn', opt=opt)
    mainQN.__init_train__(opt)
    targetQN = QNetwork(name='target_qn', opt=opt)

    update_target_op = update_target_variables(mainQN, targetQN, tau=1.0)

    logger.set_level(logger.INFO)
    env = gym.make(opt.env_id)
    env = wrappers.AtariPreprocessing(env)
    env._max_episode_steps = opt.hyper.max_steps
    # env = wrappers.Monitor(env, directory=id_path + '/video', force=True)
    env.reset()

    state = pretrain(env, opt)
    gc.collect()
    train(env, state, opt, mainQN, targetQN, update_target_op, id_path)

    env.monitor.close()
