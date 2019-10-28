import tensorflow as tf
from agents.a2c_gym.distributions import fc, make_pdtype


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

            # Calculate the loss
            # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

            # Policy loss
            self.neglogpac = train_model.pd.neglogp(A)
            # L = A(s,a) * -logpi(a|s)
            pg_loss = tf.reduce_mean(ADV * neglogpac)

            # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
            entropy = tf.reduce_mean(train_model.pd.entropy())

            # Value loss
            vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

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
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=opt.hyper.learning_rate, decay=alpha, epsilon=epsilon)

            self.train_op = self.optimizer.apply_gradients(grads)

            tf.summary.scalar('loss', self.loss)

            self.reward_summary = tf.placeholder(tf.float32, name='reward_summary')
            tf.summary.scalar('reward', self.reward_summary)




















