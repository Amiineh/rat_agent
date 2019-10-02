import tensorflow as tf


class QNetwork:

    def __init__(self, name, opt):
        with tf.variable_scope(name):
            self.name = name

            self.inputs_ = tf.placeholder(tf.float32,
                                          [None, opt.env.state_size[0], opt.env.state_size[1], opt.env.state_size[2]],
                                          name='inputs')
            self.conv1 = tf.contrib.layers.conv2d(self.inputs_, opt.hyper.output_filters_conv[0], kernel_size=8,
                                                  stride=4)
            self.conv2 = tf.contrib.layers.conv2d(self.conv1, opt.hyper.output_filters_conv[1], kernel_size=4, stride=2)
            self.conv3 = tf.contrib.layers.conv2d(self.conv2, opt.hyper.output_filters_conv[2], kernel_size=4, stride=1)

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
        self.loss = tf.reduce_mean(tf.square(self.y - batched_index(self.output, self.action)))
        self.train_op = tf.train.AdamOptimizer(opt.hyper.learning_rate).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)


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
