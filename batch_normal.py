"""
from: https://gitlab.doc.ic.ac.uk/cbaumgar/ifind1_scanplanes_tensorflow/blob/b7201525acc1f3fa462beb427845219f1859457f/batch_normal.py
and from: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
"""
import tensorflow as tf

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
            name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
            name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
            mean_var_with_update,
            lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_normalization(x, mean, var,
            beta, gamma, 1e-3)
    return normed
