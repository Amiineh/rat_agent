import tensorflow as tf


class ActorCriticNetwork:

    def __init__(self, name, opt):
        with tf.variable_scope(name):
            self.name = name

            self.inputs_ = tf.placeholder(tf.float32,
                                          [None, opt.env.state_size[0], opt.env.state_size[1], opt.env.state_size[2]],
                                          name='inputs')
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, opt.hyper.output_filters_conv[0], kernel_size=opt.hyper.kernel_size[0], stride=opt.hyper.stride[0])
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, opt.hyper.output_filters_conv[1], kernel_size=opt.hyper.kernel_size[1], stride=opt.hyper.stride[1])

            self.fc1 = tf.contrib.layers.fully_connected(
                tf.reshape(self.conv2, [-1, self.conv2.shape[1] * self.conv2.shape[2] * self.conv2.shape[3]]),
                opt.hyper.hidden_size)

            # Value function - Linear output layer
            self.value_output = tf.contrib.layers.fully_connected(self.fc1, 1,
                                                                  activation_fn=None)

            # Policy - softmax output layer
            self.policy_logits = tf.contrib.layers.fully_connected(self.fc1, opt.env.action_size, activation_fn=None)
            self.policy_output = tf.contrib.layers.softmax(self.policy_logits)

            self.action_output = tf.squeeze(tf.multinomial(logits=self.policy_logits, num_samples=1), axis=1)

            # todo:
            self.rewards = tf.placeholder(tf.float32, [opt.hyper.n, opt.hyper.num_envs], name="rewards")
            self.discounts = tf.placeholder(tf.float32, [opt.hyper.n, opt.hyper.num_envs], name="discounts")
            self.initial_Rs = tf.placeholder(tf.float32, [opt.hyper.num_envs], name="initial_Rs")

            # Used for trfl stuff
            self.value_output_unflat = tf.reshape(self.value_output, [opt.hyper.n, opt.hyper.num_envs])
            self.policy_logits_unflat = tf.reshape(self.policy_logits, [opt.hyper.n, opt.hyper.num_envs, -1])
            a2c_loss, extra = trfl.sequence_advantage_actor_critic_loss(
                policy_logits=self.policy_logits_unflat,
                baseline_values=self.value_output_unflat,
                actions=self.actions_,
                rewards=self.rewards,
                pcontinues=self.discounts,
                bootstrap_value=self.initial_Rs,
                entropy_cost=entropy_reg_term,
                normalise_entropy=normalise_entropy)
            self.loss = tf.reduce_mean(a2c_loss)
            self.extra = extra
            self.train_op = tf.train.RMSPropOptimizer(opt.hyper.learning_rate).minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

            self.reward_summary = tf.placeholder(tf.float32, name='reward_summary')
            tf.summary.scalar('reward', self.reward_summary)
