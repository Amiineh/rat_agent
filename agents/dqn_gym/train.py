import numpy as np
import time
import random
import tensorflow as tf
import trfl
import gym
from gym import wrappers, logger
from agents.dqn_gym.memory import Memory
from agents.dqn_gym.network import QNetwork

# Profiling Variables
# TODO: Make more clever profiling scheme
total_steps = 0
total_step_time = 0
mean_step_time = 0

total_network_updates = 0
total_network_update_time = 0
mean_network_update_time = 0

start_program = 0

def get_random_action():
    return random.randint(0, 3)


def index_to_english(action):
    # one can also use: env.unwrapped.get_action_meanings()
    english_names_of_actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    return english_names_of_actions[action]


def env_step(env, action, num_repeats=60):
    print(index_to_english(action))
    reward = 0
    count = 0
    while count < num_repeats:

        if not env.is_running():
            env.reset()

        # Profile environment step
        global total_steps, total_step_time, mean_step_time
        start = time.clock()
        observation, reward, done, info = env.step(action)
        step_time = time.clock() - start
        total_step_time += step_time
        total_steps += 1
        mean_step_time = total_step_time / total_steps

        if reward != 0:
            print("REWARD: " + str(reward))
            break

        count += 1

    done = reward > 0
    next_state = env.observations()['RGB_INTERLEAVED']
    # next_state = np.reshape(next_state, [-1])

    return next_state, reward, done


def pretrain(env, memory, opt):
    state, reward, done = env_step(env, get_random_action())

    # Make a bunch of random actions and store the experiences
    for ii in range(opt.hyper.pretrain_length):

        action = get_random_action()
        next_state, reward, done = env_step(env, action)

        if done:
            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step to get the pole and cart moving
            # state, reward, done = env_step(env, get_random_action())


        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state

    return state


def train(env, memory, state, opt, mainQN, targetQN, target_network_update_ops, id_path):
    # Now train with experiences
    global start_program
    start_program = time.clock()
    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(id_path, sess.graph)

        step = 0
        for ep in range(1, opt.hyper.train_episodes):

            total_program_time = time.clock() - start_program
            print("Mean step time: ", mean_step_time)
            print("Mean network update time: ", mean_network_update_time)
            print("Total step time: ", total_step_time)
            print("Total network update time: ", total_network_update_time)
            print("The rest of the program time: ", total_program_time - (total_step_time + total_network_update_time))

            total_reward = 0
            t = 0
            while t < opt.hyper.max_steps:
                step += 1

                # update target q network
                if step % opt.hyper.update_target_every == 0:
                    # TRFL way
                    sess.run(target_network_update_ops)
                    print("\nCopied model parameters to target network.")

                # Explore or Exploit
                explore_p = opt.hyper.explore_stop + \
                            (opt.hyper.explore_start - opt.hyper.explore_stop) * np.exp(-opt.hyper.decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = get_random_action()
                else:
                    #  Add profiling
                    global total_network_update_time, total_network_updates, mean_network_update_time
                    start = time.clock()

                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                    network_update_time = time.clock() - start
                    total_network_update_time += network_update_time
                    total_network_updates += 1
                    mean_network_update_time = total_network_update_time / total_network_updates

                # Take action, get new state and reward
                next_state, reward, done = env_step(env, action)
                total_reward += reward

                if done or t == opt.hyper.max_steps - 1:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = opt.hyper.max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {}'.format(total_reward),
                          # 'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(opt.hyper.batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train and profile network
                global total_network_update_time, total_network_updates, mean_network_update_time
                start = time.clock()
                target_Qs = sess.run(targetQN.output, feed_dict={targetQN.inputs_: next_states})
                network_update_time = time.clock() - start
                total_network_update_time += network_update_time
                total_network_updates += 1
                mean_network_update_time = total_network_update_time / total_network_updates

                # Set target_Qs to 0 for states where episode ends
                # TODO: This is kinda weird with the mapping.
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=(1, 2, 3))
                target_Qs[episode_ends] = 0

                # TRFL way, calculate td_error within TRFL
                # Profiling
                merge = tf.summary.merge_all()

                global total_network_update_time, total_network_updates, mean_network_update_time
                start = time.clock()

                loss, _, _ = sess.run([mainQN.loss, mainQN.opt, merge],
                                   feed_dict={mainQN.inputs_: states,
                                              mainQN.targetQs_: target_Qs,
                                              mainQN.reward: rewards,
                                              mainQN.actions_: actions})

                network_update_time = time.clock() - start
                total_network_update_time += network_update_time
                total_network_updates += 1
                mean_network_update_time = total_network_update_time / total_network_updates

                # train_writer.add_summary(summary, t)

            # print("Saving...")
            # saver.save(sess, '/mnt/hgfs/ryanprinster/lab/models/my_model', global_step=ep)

            # print("Resoring...")
            # saver.restore(sess, tf.train.latest_checkpoint('/mnt/hgfs/ryanprinster/lab/models/'))


def run(opt, id_path):
    """Spins up an environment and runs the random agent."""

    tf.reset_default_graph()
    mainQN = QNetwork(name='main_qn', opt=opt)
    targetQN = QNetwork(name='target_qn', opt=opt)

    target_network_update_ops = trfl.update_target_variables(targetQN.get_qnetwork_variables(),
                                                             mainQN.get_qnetwork_variables(), tau=1.0)

    logger.set_level(logger.INFO)
    env = gym.make(opt.env_id)
    env = wrappers.Monitor(env, directory=id_path+'/video', force=True)
    env.reset()
    memory = Memory(max_size=opt.hyper.memory_size)

    state = pretrain(env, memory, opt)
    train(env, memory, state, opt, mainQN, targetQN, target_network_update_ops, id_path)