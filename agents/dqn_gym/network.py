# class RandomAgent(object):
#
#     def __init__(self, action_space):
#         self.action_space = action_space
#
#     def act(self, observation, reward, done):
#         return self.action_space.sample()
import tensorflow as tf
import trfl


class QNetwork:

    def __init__(self, name, opt):
        with tf.variable_scope(name):
            self.name = name
            self.inputs_ = tf.placeholder(tf.float32, [None, opt.env.state_size[0], opt.env.state_size[1], opt.env.state_size[2]],
                                          name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [opt.hyper.batch_size], name='actions')
            # one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            # self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, opt.hyper.output_filters_conv1, kernel_size=8, stride=4)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, opt.hyper.output_filters_conv2, kernel_size=4, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, opt.hyper.output_filters_conv3, kernel_size=4, stride=1)

            self.fc1 = tf.contrib.layers.fully_connected(
                tf.reshape(self.conv3, [-1, self.conv3.shape[1] * self.conv3.shape[2] * self.conv3.shape[3]]),
                opt.hyper.hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc1, opt.env.action_size,
                                                            activation_fn=None)

            # tf.summary.histogram("output", self.output)

            print("Network shapes:")
            print(self.conv1.shape)
            print(self.conv2.shape)
            print(self.conv3.shape)
            print(self.fc1.shape)
            print(self.output.shape)

            # TRFL way
            self.targetQs_ = tf.placeholder(tf.float32, [opt.hyper.batch_size, opt.env.action_size], name='target')
            self.reward = tf.placeholder(tf.float32, [opt.hyper.batch_size], name="reward")
            self.discount = tf.constant(0.99, shape=[opt.hyper.batch_size], dtype=tf.float32, name="discount")

            # print(self.output.shape)
            # print(self.actions_.shape)
            # print(self.reward.shape)
            # print(self.discount.shape)
            # print(self.targetQs_.shape)
            # TRFL qlearning
            qloss, q_learning = trfl.qlearning(self.output, self.actions_, self.reward, self.discount, self.targetQs_)
            self.loss = tf.reduce_mean(qloss)
            self.opt = tf.train.AdamOptimizer(opt.hyper.learning_rate).minimize(self.loss)
            tf.summary.scalar('loss', self.loss)

    def get_qnetwork_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
