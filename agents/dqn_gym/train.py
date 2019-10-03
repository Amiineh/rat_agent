import numpy as np
import random
import tensorflow as tf
import gym
from gym import wrappers, logger
from agents.dqn_gym.memory import Memory
from agents.dqn_gym.network import QNetwork


def update_target_variables(mainQN, targetQN, tau=1.0):
    q_vars = mainQN.get_network_variables()
    q_target_vars = targetQN.get_network_variables()
    # assert len(q_vars) == len(q_target_vars)
    update_target_op = [v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q_target_vars, q_vars)]
    return update_target_op


def get_random_action():
    return random.randint(0, 3)


def index_to_english(env, action):
    # for Breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    english_names_of_actions = env.unwrapped.get_action_meanings()
    return english_names_of_actions[action]


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


def pretrain(env, memory, opt):
    state, reward, done = env_step(env, get_random_action())

    # Make a bunch of random actions and store the experiences
    for _ in range(opt.hyper.pretrain_length):

        action = get_random_action()
        next_state, reward, done = env_step(env, action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            # Start new episode
            env.reset()
        else:
            memory.add((state, action, reward, next_state, done))
            state = next_state

    return state


def train(env, memory, state, opt, mainQN, targetQN, update_target_op, id_path):
    global summary
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(id_path, sess.graph)

        step = 0
        for ep in range(1, opt.hyper.train_episodes):
            total_reward = 0
            t = 0
            while t < opt.hyper.max_steps:
                step += 1

                # update target q network
                if step % opt.hyper.update_target_every == 0:
                    sess.run(update_target_op)
                    print("\nCopied model parameters to target network.")

                # Explore or Exploit
                explore_p = opt.hyper.explore_stop + \
                            (opt.hyper.explore_start - opt.hyper.explore_stop) * np.exp(-opt.hyper.decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = get_random_action()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done = env_step(env, action)
                total_reward += reward

                # todo: change for deepmind_lab
                if done or t == opt.hyper.max_steps - 1:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = opt.hyper.max_steps
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    # Start new episode
                    env.reset()

                    print('\nEpisode: {}'.format(ep),
                          '\nTotal reward: {}'.format(total_reward),
                          '\nExplore P: {:.4f}'.format(explore_p))

                    tf.summary.scalar('reward', total_reward)
                    tf.summary.scalar('explore_p', explore_p)

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    state = next_state
                    t += 1

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
                merge = tf.summary.merge_all()

                loss, _, summary = sess.run([mainQN.loss, mainQN.train_op, merge],
                                            feed_dict={mainQN.inputs_: states,
                                                       mainQN.targetQs_: target_Qs,
                                                       mainQN.reward: rewards,
                                                       mainQN.action: actions})

            train_writer.add_summary(summary, ep)

            # print("Saving...")
            # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

            # print("Resoring...")
            # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))


def run(opt, id_path):
    tf.reset_default_graph()
    memory = Memory(max_size=opt.hyper.memory_size)
    mainQN = QNetwork(name='main_qn', opt=opt)
    mainQN.__init_train__(opt)
    targetQN = QNetwork(name='target_qn', opt=opt)

    update_target_op = update_target_variables(mainQN, targetQN, tau=1.0)

    logger.set_level(logger.INFO)
    env = gym.make(opt.env_id)
    env = wrappers.AtariPreprocessing(env)
    env._max_episode_steps = opt.hyper.max_steps
    env = wrappers.Monitor(env, directory=id_path + '/video', force=True)
    env.reset()

    state = pretrain(env, memory, opt)
    train(env, memory, state, opt, mainQN, targetQN, update_target_op, id_path)

    env.monitor.close()
