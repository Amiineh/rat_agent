import tensorflow as tf


class QNetwork:

    def __init__(self, name, opt):
        with tf.variable_scope(name):
            self.name = name

            self.inputs_ = tf.placeholder(tf.float32,
                                          [None, opt.env.state_size[0], opt.env.state_size[1], opt.env.state_size[2]],
                                          name='inputs')
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, opt.hyper.output_filters_conv[0], kernel_size=opt.hyper.kernel_size[0], stride=opt.hyper.stride[0],
                                                  weights_initializer=tf.variance_scaling_initializer(scale=1/(opt.hyper.kernel_size[0]**2)))
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, opt.hyper.output_filters_conv[1], kernel_size=opt.hyper.kernel_size[1], stride=opt.hyper.stride[1],
                                                  weights_initializer=tf.variance_scaling_initializer(scale=1/(opt.hyper.kernel_size[1] ** 2)))
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, opt.hyper.output_filters_conv[2], kernel_size=opt.hyper.kernel_size[2], stride=opt.hyper.stride[2],
                                                  weights_initializer=tf.variance_scaling_initializer(scale=1/(opt.hyper.kernel_size[2] ** 2)))

            self.fc1 = tf.contrib.layers.fully_connected(
                tf.reshape(self.conv3, [-1, self.conv3.shape[1] * self.conv3.shape[2] * self.conv3.shape[3]]),
                opt.hyper.hidden_size)

            self.output = tf.contrib.layers.fully_connected(
                self.fc1, opt.env.action_size, activation_fn=None)

    def get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def __init_train__(self, opt):
        self.targetQs_ = tf.placeholder(tf.float32, [opt.hyper.batch_size, opt.env.action_size], name='target')
        self.reward = tf.placeholder(tf.float32, [opt.hyper.batch_size], name='reward')
        self.action = tf.placeholder(tf.int32, [opt.hyper.batch_size], name='action')

        self.y = self.reward + opt.hyper.gamma * tf.reduce_max(self.targetQs_, axis=1)
        self.error = self.y - batched_index(self.output, self.action)
        self.loss = tf.reduce_mean(huber_loss(self.error))
        self.optimizer = tf.train.RMSPropOptimizer(opt.hyper.learning_rate)

        gradients = self.optimizer.compute_gradients(self.loss, var_list=self.get_network_variables())
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 1.), var)
        self.train_op = self.optimizer.apply_gradients(gradients)

        tf.summary.scalar('loss', self.loss)

        self.reward_summary = tf.placeholder(tf.float32, name='reward_summary')
        tf.summary.scalar('reward', self.reward_summary)


# from trfl:
# https://github.com/deepmind/trfl/blob/master/trfl/indexing_ops.py
def batched_index(values, indices):
    """Equivalent to `values[:, indices]`.
    Performs indexing on batches and sequence-batches by reducing over
    zero-masked values. Compared to indexing with `tf.gather` this approach is
    more general and TPU-friendly, but may be less efficient if `num_values`
    is large. It works with tensors whose shapes are unspecified or
    partially-specified, but this op will only do shape checking on shape
    information available at graph construction time. When complete shape
    information is absent, certain shape incompatibilities may not be detected at
    runtime! See `indexing_ops_test` for detailed examples.
    Args:
    values: tensor of shape `[B, num_values]` or `[T, B, num_values]`
    indices: tensor of shape `[B]` or `[T, B]` containing indices.
    Returns:
    Tensor of shape `[B]` or `[T, B]` containing values for the given indices.
    Raises: ValueError if values and indices have sizes that are known
    statically (i.e. during graph construction), and those sizes are not
    compatible (see shape descriptions in Args list above).
    """
    with tf.name_scope("batch_indexing", values=[values, indices]):
        one_hot_indices = tf.one_hot(
            indices, tf.shape(values)[-1], dtype=values.dtype)
        return tf.reduce_sum(values * one_hot_indices, axis=-1)


# # from trfl
# # https://github.com/deepmind/trfl/blob/master/trfl/clipping_ops.py
# def huber_loss(input_tensor, quadratic_linear_boundary, name=None):
#   """Calculates huber loss of `input_tensor`.
#   For each value x in `input_tensor`, the following is calculated:
#   ```
#     0.5 * x^2                  if |x| <= d
#     0.5 * d^2 + d * (|x| - d)  if |x| > d
#   ```
#   where d is `quadratic_linear_boundary`.
#   When `input_tensor` is a loss this results in a form of gradient clipping.
#   This is, for instance, how gradients are clipped in DQN and its variants.
#   Args:
#     input_tensor: `Tensor`, input values to calculate the huber loss on.
#     quadratic_linear_boundary: `float`, the point where the huber loss function
#       changes from a quadratic to linear.
#     name: `string`, name for the operation (optional).
#   Returns:
#     `Tensor` of the same shape as `input_tensor`, containing values calculated
#     in the manner described above.
#   Raises:
#     ValueError: if quadratic_linear_boundary <= 0.
#   """
#   if quadratic_linear_boundary < 0:
#     raise ValueError("quadratic_linear_boundary must be > 0.")
#
#   with tf.name_scope(
#       name, default_name="huber_loss",
#       values=[input_tensor, quadratic_linear_boundary]):
#     abs_x = tf.abs(input_tensor)
#     delta = quadratic_linear_boundary
#     quad = tf.minimum(abs_x, delta)
#     # The following expression is the same in value as
#     # tf.maximum(abs_x - delta, 0), but importantly the gradient for the
#     # expression when abs_x == delta is 0 (for tf.maximum it would be 1). This
#     # is necessary to avoid doubling the gradient, since there is already a
#     # non-zero contribution to the gradient from the quadratic term.
#     lin = (abs_x - quad)
#     return 0.5 * quad**2 + delta * lin


# baselines
# https://github.com/openai/baselines/blob/1fb4dfb7802083de13567ac567c84e7dfd7d1514/baselines/common/tf_util.py
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )