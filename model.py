# coding=utf-8
import tensorflow as tf
import re
import numpy as np
from config_loader import globals as g_
import batch_normal as bn

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', 0.000001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('margin', g_.CONTRASTIVE_LOSS_MARGIN,
                          """contrastive loss(l2) margin""")
tf.app.flags.DEFINE_boolean('mvcnn_shared', g_.MVCNN_SHARED,
                            """mvcnn 12-streams weights shared or not.""")
tf.app.flags.DEFINE_boolean('cd_shared', g_.CROSS_DOMAIN_SHARED,
                            """share weights between 2 domains CNNs""")
tf.app.flags.DEFINE_boolean(
    'cd_adaptation', g_.CROSS_DOMAIN_ADAPTATION,
    """adaptation layer (FC + ReLU) after feature of image stream.""")
tf.app.flags.DEFINE_string('layer', g_.FEATURE_LAYER,
                           """output featuremap layer for representation""")
tf.app.flags.DEFINE_boolean('conv4_pool5_concat', g_.CONV4_POOL5_CONCAT,
                            """[conv4,pool5] concate to combine appearance and semantic similarity
This can be only used with "output_layer" = pool5, {"mvcnn_shared","cd_shared","cd_adaptation"} are false.
                            """)
tf.app.flags.DEFINE_string(
    'convfc_initialize',
    g_.CONVFC_INIT,
    """convlstm initializer, can be "orthogonal" or "xavier", default is % s""" %
    g_.CONVFC_INIT)
tf.app.flags.DEFINE_boolean(
    'fuse_3dconv', g_.FUSE_3DCONV,
    """3D CONV replacing view-pooling for sequential view inputs""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# total_loss = loss + WD_FACTOR * (weight sum), to make the term 3500 into 0.02
WEIGHT_DECAY_FACTOR = 5e-6

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'


def _conv(name, phase_train, in_, ksize, strides=[1, 1, 1, 1],
          padding=DEFAULT_PADDING,
          batch_norm=False,
          reuse=False,
          group=1):

    n_kern = ksize[3]

    with tf.variable_scope(name, reuse=reuse) as scope:

        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = tf.nn.conv2d(in_, kernel, strides, padding=padding)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=in_)
            kernel_groups = tf.split(
                axis=3, num_or_size_splits=group, value=kernel)

            def convolve(
                i, k): return tf.nn.conv2d(
                i, k, strides, padding=padding)
            output_groups = [
                convolve(
                    i, k) for i, k in zip(
                    input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(axis=3, values=output_groups)

        biases = _variable_on_cpu(
            'biases', [n_kern],
            tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        if g_.BATCH_NORM:
            n_out = ksize[-1]
            if g_.BN_AFTER_ACTV:
                conv = tf.nn.relu(conv, name=scope.name)
                conv = bn.batch_norm(conv, n_out, phase_train)
            else:
                conv = bn.batch_norm(conv, n_out, phase_train)
                conv = tf.nn.relu(conv, name=scope.name)
        else:
            conv = tf.nn.relu(conv, name=scope.name)

        _activation_summary(conv)

    print name, conv.get_shape().as_list()
    return conv


def _3dconv(
        name,
        in_,
        ksize,
        strides=[
            1,
            1,
            1,
            1,
            1],
    padding=DEFAULT_PADDING,
        reuse=False):

    n_kern = ksize[4]

    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
        conv = tf.nn.conv3d(in_, kernel, strides, padding=padding)
        biases = _variable_on_cpu(
            'biases', [n_kern],
            tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)

    print name, conv.get_shape().as_list()
    return conv


def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print name, pool.get_shape().as_list()
    return pool


def _fc(name, in_, outsize, dropout=1.0):
    with tf.variable_scope(name, reuse=False) as scope:
        # Move everything into depth so we can perform a single matrix multiply.

        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay(
            'weights', shape=[insize, outsize],
            wd=WEIGHT_DECAY_FACTOR)
        biases = _variable_on_cpu(
            'biases', [outsize],
            tf.constant_initializer(0.0))
        fc = tf.matmul(in_, weights) + biases
        fc = tf.nn.relu(fc, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)
        _activation_summary(fc)

    print name, fc.get_shape().as_list()
    return fc


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0)  # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat(axis=0, values=[vp, v])
    print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_max(vp, [0], name=name)
    return vp


def inference_multiview(
        views,
        keep_prob,
        output_layer,
        phase_train,
        reuse=False):
    """
    views: N x V x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    V = g_.V

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in xrange(V):
        p = '_view%d' % i
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv(
            'v_conv1' + p, phase_train, view, [11, 11, 3, 96],
            [1, 4, 4, 1],
            'VALID', batch_norm=g_.BATCH_NORM, reuse=reuse)
        pool1 = _maxpool(
            'v_pool1' + p, conv1, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID')

        conv2 = _conv(
            'v_conv2' + p, phase_train, pool1, [5, 5, 96, 256],
            batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
        pool2 = _maxpool(
            'v_pool2' + p, conv2, ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID')

        conv3 = _conv(
            'v_conv3' + p, phase_train, pool2, [3, 3, 256, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse)
        conv4 = _conv(
            'v_conv4' + p, phase_train, conv3, [3, 3, 384, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)

        if output_layer == 'conv4':
            output = conv4

        elif output_layer == 'pool5':
            conv5 = _conv(
                'v_conv5' + p, phase_train, conv4, [3, 3, 384, 256],
                batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
            pool5 = _maxpool(
                'v_pool5' + p, conv5, ksize=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='VALID')
            output = pool5

            if FLAGS.conv4_pool5_concat:
                conv4_1d_norm = tf.nn.l2_normalize(
                    _featuremap_to_1d(conv4), dim=1)
                pool5_1d_norm = tf.nn.l2_normalize(
                    _featuremap_to_1d(pool5), dim=1)
                output = tf.concat(
                    axis=1, values=[
                        conv4_1d_norm, pool5_1d_norm])

        reshape = _featuremap_to_1d(output)
        view_pool.append(reshape)

    vp = _view_pool(view_pool, 'v_vp')
    print 'vp', vp.get_shape().as_list()

    # fc6 = _fc('v_fc6', pool5_vp, 4096, dropout=keep_prob)
    # fc7 = _fc('v_fc7', fc6, 4096, dropout=keep_prob)
    # fc8 = _fc('v_fc8', fc7, 40)

    output = None
    if output_layer in ['conv4', 'pool5']:
        output = vp
    elif output_layer == 'fc6':
        fc6 = _fc('v_fc6', vp, 4096, dropout=keep_prob)
        output = fc6
    elif output_layer == 'fc7':
        fc6 = _fc('v_fc6', vp, 4096, dropout=keep_prob)
        fc7 = _fc('v_fc7', fc6, 4096, dropout=keep_prob)
        output = fc7

    assert output is not None, 'No matching extract_layer:' + output_layer

    # tail layer to reduce output dim
    if g_.TAIL_LAYER:
        output = _fc('v_tail', output, g_.TAIL_LAYER_DIM, dropout=keep_prob)

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def inference_multiview_mvshared(
        views,
        keep_prob,
        output_layer,
        phase_train,
        reuse=False):
    """
    views: N x V x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    V = g_.V

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in xrange(V):
        reuse = (reuse or (i != 0))
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv(
            'v_conv1', phase_train, view, [
                11, 11, 3, 96], [
                1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM, reuse=reuse)
        pool1 = _maxpool(
            'v_pool1', conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv2 = _conv(
            'v_conv2', phase_train, pool1, [5, 5, 96, 256],
            batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
        pool2 = _maxpool(
            'v_pool2', conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv3 = _conv(
            'v_conv3', phase_train, pool2, [3, 3, 256, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse)
        conv4 = _conv(
            'v_conv4', phase_train, conv3, [3, 3, 384, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)

        if output_layer == 'conv4':
            output = conv4

        elif output_layer == 'pool5':
            conv5 = _conv(
                'v_conv5', phase_train, conv4, [3, 3, 384, 256],
                batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
            pool5 = _maxpool(
                'v_pool5', conv5, ksize=[
                    1, 3, 3, 1], strides=[
                    1, 2, 2, 1], padding='VALID')
            output = pool5

        dim = 1
        for d in output.get_shape().as_list()[1:]:
            dim *= d

        reshape = tf.reshape(output, [-1, dim])

        view_pool.append(reshape)

    vp = _view_pool(view_pool, 'v_vp')
    print 'vp', vp.get_shape().as_list()

    # fc6 = _fc('v_fc6', pool5_vp, 4096, dropout=keep_prob)
    # fc7 = _fc('v_fc7', fc6, 4096, dropout=keep_prob)
    # fc8 = _fc('v_fc8', fc7, 40)

    output = None
    if output_layer in ['conv4', 'pool5']:
        output = vp
    elif output_layer == 'fc6':
        fc6 = _fc('v_fc6', vp, 4096, dropout=keep_prob)
        output = fc6
    elif output_layer == 'fc7':
        fc6 = _fc('v_fc6', vp, 4096, dropout=keep_prob)
        fc7 = _fc('v_fc7', fc6, 4096, dropout=keep_prob)
        output = fc7

    assert output is not None, 'No matching extract_layer:' + output_layer

    # tail layer to reduce output dim
    if g_.TAIL_LAYER:
        output = _fc('v_tail', output, g_.TAIL_LAYER_DIM, dropout=keep_prob)

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def inference_multiview_mvshared_cdshared(
        views, keep_prob, output_layer, phase_train):
    """
    multi-view shared, cross-domain shared
    views: N x V x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    V = g_.V

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in xrange(V):
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv(
            'i_conv1', phase_train, view, [
                11, 11, 3, 96], [
                1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM, reuse=True)
        pool1 = _maxpool(
            'i_pool1', conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv2 = _conv(
            'i_conv2', phase_train, pool1, [5, 5, 96, 256],
            batch_norm=g_.BATCH_NORM, reuse=True, group=2)
        pool2 = _maxpool(
            'i_pool2', conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv3 = _conv(
            'i_conv3', phase_train, pool2, [3, 3, 256, 384],
            batch_norm=g_.BATCH_NORM, reuse=True)
        conv4 = _conv(
            'i_conv4', phase_train, conv3, [3, 3, 384, 384],
            batch_norm=g_.BATCH_NORM, reuse=True, group=2)

        if output_layer == 'conv4':
            output = conv4

        elif output_layer == 'pool5':
            conv5 = _conv(
                'i_conv5', phase_train, conv4, [3, 3, 384, 256],
                batch_norm=g_.BATCH_NORM, reuse=True, group=2)
            pool5 = _maxpool(
                'i_pool5', conv5, ksize=[
                    1, 3, 3, 1], strides=[
                    1, 2, 2, 1], padding='VALID')
            output = pool5

        dim = 1
        for d in output.get_shape().as_list()[1:]:
            dim *= d

        reshape = tf.reshape(output, [-1, dim])

        view_pool.append(reshape)

    vp = _view_pool(view_pool, 'v_vp')
    print 'vp', vp.get_shape().as_list()

    # fc6 = _fc('v_fc6', pool5_vp, 4096, dropout=keep_prob)
    # fc7 = _fc('v_fc7', fc6, 4096, dropout=keep_prob)
    # fc8 = _fc('v_fc8', fc7, 40)

    output = None
    if output_layer in ['conv4', 'pool5']:
        output = vp
    elif output_layer == 'fc6':
        fc6 = _fc('i_fc6', vp, 4096, dropout=keep_prob)
        output = fc6
    elif output_layer == 'fc7':
        fc6 = _fc('i_fc6', vp, 4096, dropout=keep_prob)
        fc7 = _fc('i_fc7', fc6, 4096, dropout=keep_prob)
        output = fc7

    assert output is not None, 'No matching extract_layer:' + output_layer

    # tail layer to reduce output dim
    if g_.TAIL_LAYER:
        output = _fc('v_tail', output, g_.TAIL_LAYER_DIM, dropout=keep_prob)

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def inference_multiview_mvshared_cdshared_3dconv(
        views, keep_prob, output_layer, phase_train, reuse=False):
    """
    multi-view shared, cross-domain shared
    views: N x V x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    V = g_.V
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in xrange(V):
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv(
            'i_conv1', phase_train, view, [
                11, 11, 3, 96], [
                1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM, reuse=True)
        pool1 = _maxpool(
            'i_pool1', conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv2 = _conv(
            'i_conv2', phase_train, pool1, [5, 5, 96, 256],
            batch_norm=g_.BATCH_NORM, reuse=True, group=2)
        pool2 = _maxpool(
            'i_pool2', conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv3 = _conv(
            'i_conv3', phase_train, pool2, [3, 3, 256, 384],
            batch_norm=g_.BATCH_NORM, reuse=True)
        conv4 = _conv(
            'i_conv4', phase_train, conv3, [3, 3, 384, 384],
            batch_norm=g_.BATCH_NORM, reuse=True, group=2)

        conv5 = _conv(
            'i_conv5', phase_train, conv4, [3, 3, 384, 256],
            batch_norm=g_.BATCH_NORM, reuse=True, group=2)
        pool5 = _maxpool(
            'pool5', conv5, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')
        view_pool.append(pool5)

    view_pool = tf.transpose(view_pool, perm=[1, 0, 2, 3, 4])
    conv6 = _3dconv(
        '3dconv6', view_pool, [12, 1, 1, 256, 256],
        padding='VALID', reuse=reuse)
    feature = conv6

    dim = np.prod(feature.get_shape().as_list()[1:])
    reshape = tf.reshape(feature, [-1, dim])

    assert output_layer in [
        'pool5'], '3dconv only supports output = conv5 / pool5'
    output = reshape

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def inference_multiview_mvshared_3dconv(
        views,
        keep_prob,
        output_layer,
        phase_train,
        reuse=False):
    """
    multi-view shared, no cross-domain sharing
    views: N x V x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    V = g_.V
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])

    view_pool = []
    for i in xrange(V):
        reuse_2 = (reuse or (i != 0))
        view = tf.gather(views, i)  # NxWxHxC

        conv1 = _conv(
            'v_conv1', phase_train, view, [
                11, 11, 3, 96], [
                1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM, reuse=reuse_2)
        pool1 = _maxpool(
            'v_pool1', conv1, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv2 = _conv(
            'v_conv2', phase_train, pool1, [5, 5, 96, 256],
            batch_norm=g_.BATCH_NORM, reuse=reuse_2, group=2)
        pool2 = _maxpool(
            'v_pool2', conv2, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')

        conv3 = _conv(
            'v_conv3', phase_train, pool2, [3, 3, 256, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse_2)
        conv4 = _conv(
            'v_conv4', phase_train, conv3, [3, 3, 384, 384],
            batch_norm=g_.BATCH_NORM, reuse=reuse_2, group=2)

        conv5 = _conv(
            'v_conv5', phase_train, conv4, [3, 3, 384, 256],
            batch_norm=g_.BATCH_NORM, reuse=reuse_2, group=2)
        pool5 = _maxpool(
            'v_pool5', conv5, ksize=[
                1, 3, 3, 1], strides=[
                1, 2, 2, 1], padding='VALID')
        view_pool.append(pool5)

    view_pool = tf.transpose(view_pool, perm=[1, 0, 2, 3, 4])
    conv6 = _3dconv(
        'v_3dconv6', view_pool, [12, 1, 1, 256, 256],
        padding='VALID', reuse=reuse)
    feature = conv6

    dim = np.prod(feature.get_shape().as_list()[1:])
    reshape = tf.reshape(feature, [-1, dim])

    assert output_layer in [
        'pool5'], '3dconv only supports output = conv5 / pool5'
    output = reshape

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def inference_image(image, keep_prob, output_layer, phase_train, reuse=False):
    """
    images: N x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    extract_layer: name of the layer which will be output, {'pool5', 'fc6', 'fc7'}
    """

    conv1 = _conv(
        'i_conv1', phase_train, image, [
            11, 11, 3, 96], [
            1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM, reuse=reuse)
    lrn1 = None
    pool1 = _maxpool(
        'i_pool1', conv1, ksize=[
            1, 3, 3, 1], strides=[
            1, 2, 2, 1], padding='VALID')

    conv2 = _conv(
        'i_conv2', phase_train, pool1, [5, 5, 96, 256],
        batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
    lrn2 = None
    pool2 = _maxpool(
        'i_pool2', conv2, ksize=[
            1, 3, 3, 1], strides=[
            1, 2, 2, 1], padding='VALID')

    conv3 = _conv(
        'i_conv3', phase_train, pool2, [3, 3, 256, 384],
        batch_norm=g_.BATCH_NORM, reuse=reuse)
    conv4 = _conv(
        'i_conv4', phase_train, conv3, [3, 3, 384, 384],
        batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)

    print output_layer
    if output_layer == 'conv4':
        feature = conv4
    elif output_layer in ['conv5', 'pool5']:
        conv5 = _conv(
            'i_conv5', phase_train, conv4, [3, 3, 384, 256],
            batch_norm=g_.BATCH_NORM, reuse=reuse, group=2)
        if output_layer == 'conv5':
            feature = conv5
        else:
            pool5 = _maxpool(
                'i_pool5', conv5, ksize=[
                    1, 3, 3, 1], strides=[
                    1, 2, 2, 1], padding='VALID')
            feature = pool5

            if FLAGS.conv4_pool5_concat:
                conv4_1d_norm = tf.nn.l2_normalize(
                    _featuremap_to_1d(conv4), dim=1)
                pool5_1d_norm = tf.nn.l2_normalize(
                    _featuremap_to_1d(pool5), dim=1)
                feature = tf.concat(
                    axis=1, values=[
                        conv4_1d_norm, pool5_1d_norm])

    feature = _featuremap_to_1d(feature)

    # fc6 = _fc('i_fc6', pool5, 4096, dropout=keep_prob)
    # fc7 = _fc('i_fc7', fc6, 4096, dropout=keep_prob)
    # fc8 = _fc('i_fc8', fc7, 40)

    output = None
    if output_layer in ['conv4', 'conv5', 'pool5']:
        output = feature
    elif output_layer == 'fc6':
        fc6 = _fc('i_fc6', pool5, 4096, dropout=keep_prob)
        output = fc6
    elif output_layer == 'fc7':
        fc6 = _fc('i_fc6', pool5, 4096, dropout=keep_prob)
        fc7 = _fc('i_fc7', fc6, 4096, dropout=keep_prob)
        output = fc7

    assert output is not None, 'No matching extract_layer:' + output_layer

    # adaptation layer after feature of image
    if FLAGS.cd_adaptation:
        output = tf.nn.l2_normalize(
            output, dim=1, name='normalized_feature_before_adapt')
        output = _fc(
            'adaptation_fc',
            output,
            output.get_shape().as_list()[1],
            dropout=keep_prob)

    # tail layer to reduce output dim
    if g_.TAIL_LAYER:
        output = _fc('i_tail', output, g_.TAIL_LAYER_DIM, dropout=keep_prob)

    # L2-nomalization
    if g_.L2_NORMALIZATION:
        output = tf.nn.l2_normalize(output, dim=1, name='normalized_feature')

    return output


def _featuremap_to_1d(feature):
    """
    convert a (N x D1 x D2 x..x Dk) tensor to a (N x M) tensor,
    where M = D1 x D2 .. x Dk, k can be 1 to any integer
    """
    dim = np.prod(feature.get_shape().as_list()[1:])
    return tf.reshape(feature, [-1, dim])


def load_alexnet_to_mvcnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """
    V = g_.V

    caffemodel = np.load(caffetf_modelpath, allow_pickle=True)
    data_dict = caffemodel.item()
    for v in xrange(V):
        for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            name = 'v_' + l + '_view%d' % v
            _load_param(sess, name, data_dict[l])

    for l in ['fc6', 'fc7']:
        _load_param(sess, 'v_' + l, data_dict[l])


def load_alexnet_to_mvcnn_shared(sess, caffetf_modelpath):
    """ caffemodel: np.array, """
    V = g_.V

    caffemodel = np.load(caffetf_modelpath, allow_pickle=True)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        name = 'v_' + l
        _load_param(sess, name, data_dict[l])

    for l in ['fc6', 'fc7']:
        _load_param(sess, 'v_' + l, data_dict[l])


def load_alexnet_to_icnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath, allow_pickle=True)
    data_dict = caffemodel.item()

    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = 'i_' + l
        _load_param(sess, name, data_dict[l])


def _load_param(sess, name, layer_data):
    w, b = layer_data

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            print 'loading ', name, subkey

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e:
                print 'varirable not found in graph:', subkey


def forward_image(images, keep_prob, output_layer, phase_train):
    feature_i = inference_image(images, keep_prob, output_layer, phase_train)
    return feature_i


def forward_shape(views_pos, keep_prob, output_layer, phase_train):
    if FLAGS.mvcnn_shared:
        if FLAGS.cd_shared:
            if FLAGS.fuse_3dconv:
                print 'mvcnn cdshared fuse_3dconv'
                feature_v = inference_multiview_mvshared_cdshared_3dconv(
                    views_pos, keep_prob, output_layer, phase_train)
            else:
                print 'mvcnn cdshared'
                feature_v = inference_multiview_mvshared_cdshared(
                    views_pos, keep_prob, output_layer, phase_train)
        else:
            if FLAGS.fuse_3dconv:
                print 'mvcnn fuse_3dconv'
                feature_v = inference_multiview_mvshared_3dconv(
                    views_pos, keep_prob, output_layer, phase_train)
            else:
                assert False, 'nooooooo'
    else:
        assert False, 'noooooooooooooo'
    return feature_v


def loss(
        feature_i,
        feature_v,
        label_i,
        n_batch_images,
        n_shape_per_class,
        n_classes):
    if n_shape_per_class == 1:
        print '在算l2diff 做reduce_sum的reduction_indices可能要改一下'

    n_pos = n_shape_per_class
    n_neg = n_shape_per_class * (n_classes - 1)

    # 用tensor做pos_inds和neg_inds太難，乾脆生好所有index list，到時候看圖片是哪一類直接拿來用
    # index_list = []
    pos_indices_list = []
    neg_indices_list = []
    for c in xrange(n_classes):
        shape_start = c * n_shape_per_class
        shape_end = shape_start + n_shape_per_class
        pos_inds = range(shape_start, shape_end)
        neg_inds = sorted(
                          list(
                              set(range(n_classes * n_shape_per_class)) -
                              set(pos_inds)))
        pos_indices_list.append(pos_inds)
        neg_indices_list.append(neg_inds)
        # index_list.append(pos_inds + neg_inds)

    # index_list_tensor = tf.constant(index_list, dtype='int')
    # pos_indices: shape: [n_classes, n_shape_per_class]
    pos_indices_of_classes = tf.constant(
        pos_indices_list, dtype=tf.int32, name='pos_indices')
    # neg_indices: shape: [n_classes, n_shape_per_class*(n_classes-1)]
    neg_indices_of_classes = tf.constant(
        neg_indices_list, dtype=tf.int32, name='neg_indices')

    loss_pool = []
    for i in xrange(n_batch_images):
        clz = tf.cast(label_i[i], tf.int32)
        pos_indices = pos_indices_of_classes[clz]  # must I use tf.gather() ?
        neg_indices = neg_indices_of_classes[clz]

        # get features of images and shapes
        im = feature_i[i]
        pos_shs = tf.gather(feature_v, pos_indices)
        neg_shs = tf.gather(feature_v, neg_indices)

        # shape: [n_pos]
        pos_l2sq_diffs = tf.reduce_sum(((im - pos_shs) ** 2), axis=1)
        # shape: [n_neg]
        neg_l2sq_diffs = tf.reduce_sum(((im - neg_shs) ** 2), axis=1)

        pos_l2sq_diffs_mat = tf.tile(
            tf.reshape(
                pos_l2sq_diffs, (n_pos, 1)), [
                1, n_neg])
        neg_l2sq_diffs_mat = tf.tile(
            tf.reshape(
                neg_l2sq_diffs, (1, n_neg)), [
                n_pos, 1])

        margin = tf.constant(FLAGS.margin)
        losses = tf.nn.relu(pos_l2sq_diffs_mat - neg_l2sq_diffs_mat + margin)
        loss_pool.append(losses)

    all_losses = tf.stack(loss_pool)
    nonzero = tf.cast(
        tf.not_equal(
            all_losses,
            tf.constant(
                0,
                dtype=tf.float32)),
        tf.float32)
    nonzero_count = tf.reduce_sum(nonzero)

    tf.summary.scalar('nonzero_count', nonzero_count)

    # we must take care of the case that no triplet breaks the constraint, nonzero_count=0
    # if nonzero_count == 0, then just return 0.0 as the loss
    def mean_loss():
        return tf.div(tf.reduce_sum(all_losses), nonzero_count, 'triplet_loss')

    loss_mean = tf.cond(tf.greater(nonzero_count, tf.constant(0.0)),
                        lambda: mean_loss(),
                        lambda: tf.constant(0.0))

    # loss_mean = tf.div(tf.reduce_sum(all_losses), nonzero_count, 'triplet_loss')

    # print 'loss mean shape', loss_mean.get_shape()
    tf.add_to_collection('losses', loss_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def loss_shape2vec(
        feature_i,
        feature_v,
        label_i,
        n_batch_images,
        n_shape_per_class,
        n_classes):
    """
    一次抽 I 張圖，都抽不同 class 的。
    對應每張圖，都抽 C 個 shape，所以會有 I * C 個 shape。
    所以會有這樣的對應：
    圖          0      1        2        3      ..
    p shape   0:C-1  C:2C-1  2C:3C-1  3C:4C-1  .. .
    """

    if n_shape_per_class == 1:
        print '在算l2diff 做reduce_sum的reduction_indices可能要改一下'

    n_pos = n_shape_per_class
    n_neg = n_shape_per_class * (n_classes - 1)

    # 用tensor做pos_inds和neg_inds太難，乾脆生好所有index list，到時候看圖片是哪一類直接拿來用
    # index_list = []
    pos_indices_list = []
    neg_indices_list = []
    for c in xrange(n_classes):
        shape_start = c * n_shape_per_class
        shape_end = shape_start + n_shape_per_class
        pos_inds = range(shape_start, shape_end)
        neg_inds = sorted(
                          list(
                              set(range(n_classes * n_shape_per_class)) -
                              set(pos_inds)))
        pos_indices_list.append(pos_inds)
        neg_indices_list.append(neg_inds)
        # index_list.append(pos_inds + neg_inds)

    # index_list_tensor = tf.constant(index_list, dtype='int')
    # pos_indices: shape: [n_classes, n_shape_per_class]
    pos_indices_of_classes = tf.constant(
        pos_indices_list, dtype=tf.int32, name='pos_indices')
    # neg_indices: shape: [n_classes, n_shape_per_class*(n_classes-1)]
    neg_indices_of_classes = tf.constant(
        neg_indices_list, dtype=tf.int32, name='neg_indices')

    loss_pool = []
    for i in xrange(n_batch_images):
        # clz = tf.cast(label_i[i], tf.int32)
        # pos_indices = pos_indices_of_classes[clz] # must I use tf.gather() ?
        # neg_indices = neg_indices_of_classes[clz]
        pos_indices = pos_indices_of_classes[i]
        neg_indices = neg_indices_of_classes[i]

        # get features of images and shapes
        im = feature_i[i]
        pos_shs = tf.gather(feature_v, pos_indices)
        neg_shs = tf.gather(feature_v, neg_indices)

        # shape: [n_pos]
        pos_l2sq_diffs = tf.reduce_sum(((im - pos_shs) ** 2), axis=1)
        # shape: [n_neg]
        neg_l2sq_diffs = tf.reduce_sum(((im - neg_shs) ** 2), axis=1)

        pos_l2sq_diffs_mat = tf.tile(
            tf.reshape(
                pos_l2sq_diffs, (n_pos, 1)), [
                1, n_neg])
        neg_l2sq_diffs_mat = tf.tile(
            tf.reshape(
                neg_l2sq_diffs, (1, n_neg)), [
                n_pos, 1])

        margin = tf.constant(FLAGS.margin)
        losses = tf.nn.relu(pos_l2sq_diffs_mat - neg_l2sq_diffs_mat + margin)
        loss_pool.append(losses)

    all_losses = tf.stack(loss_pool)
    nonzero = tf.cast(
        tf.not_equal(
            all_losses,
            tf.constant(
                0,
                dtype=tf.float32)),
        tf.float32)
    nonzero_count = tf.reduce_sum(nonzero)

    tf.summary.scalar('nonzero_count', nonzero_count)

    # we must take care of the case that no triplet breaks the constraint, nonzero_count=0
    # if nonzero_count == 0, then just return 0.0 as the loss
    def mean_loss():
        return tf.div(tf.reduce_sum(all_losses), nonzero_count, 'triplet_loss')

    loss_mean = tf.cond(tf.greater(nonzero_count, tf.constant(0.0)),
                        lambda: mean_loss(),
                        lambda: tf.constant(0.0))

    # loss_mean = tf.div(tf.reduce_sum(all_losses), nonzero_count, 'triplet_loss')

    # print 'loss mean shape', loss_mean.get_shape()
    tf.add_to_collection('losses', loss_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def test_loss_np(feature_i, feature_v_pos, feature_v_neg):
    """loss function for validation or testing. Inputs are numpy arrays, not tensors."""

    l2sq_diff_pos = np.sum((feature_i - feature_v_pos) ** 2, axis=1)
    l2sq_diff_neg = np.sum((feature_i - feature_v_neg) ** 2, axis=1)
    margin = FLAGS.margin
    loss = l2sq_diff_pos - l2sq_diff_neg + margin
    loss_mean = np.mean(loss)

    return loss_mean


def _triplet_loss(diffs_pos, diffs_neg):
    margin = tf.constant(FLAGS.margin)
    loss = tf.square(diffs_pos) - tf.square(diffs_neg) + margin
    return loss


def _contrastive_loss(diffs, labels):
    alpha = 5.
    beta = 10.
    gamma = -2.77/10.
    batch = FLAGS.batch_size
    ones = tf.ones([batch], dtype='float32')
    labels = tf.to_float(labels)
    match_loss = alpha * tf.square(diffs, 'match_term')
    mismatch_loss = beta * tf.exp(gamma*diffs, 'mismatch_term')

    loss = tf.add_n(
        [tf.multiply(
             tf.add_n([ones, -labels]),
             match_loss, 'loss_match_mul'),
         tf.multiply(labels, mismatch_loss, 'loss_mismatch_mul')],
        'loss_add')
    print 'diffs:', diffs.get_shape(), '_contrastive_loss:', loss.get_shape()
    print 'labels:', labels.get_shape()
    print 'ones:', ones.get_shape()
    print 'shape00', (tf.add_n([ones, -labels])).get_shape()
    print 'shape01', tf.multiply(tf.add_n([ones, -labels]), match_loss).get_shape()
    print 'shape:', loss.get_shape()
    return loss


def _contrastive_loss_l2ver(l2diffs, labels):
    print '(contrastive loss l2ver), margin=', FLAGS.margin
    margin = tf.constant(FLAGS.margin)
    tf.summary.scalar('contrastive loss margin', margin)

    print 'ggininder', labels.get_shape()

    labels = tf.to_float(labels)
    match_loss = 0.5 * tf.square(l2diffs, 'match_term')
    mismatch_loss = 0.5 * tf.square(tf.maximum(0.,
                                               tf.subtract(margin, l2diffs)), 'mismatch_term')

    loss = tf.add_n(
        [tf.multiply((1 - labels),
                     match_loss, 'loss_match_mul'),
         tf.multiply(labels, mismatch_loss, 'loss_mismatch_mul')],
        'loss_add')
    print 'diffs:', l2diffs.get_shape(), '_contrastive_loss:', loss.get_shape()
    print 'match_loss shape:', match_loss.get_shape()
    print 'mismatch loss, shape:', mismatch_loss.get_shape()
    print 'labels:', labels.get_shape()
    print 'shape:', loss.get_shape()
    print loss
    return loss


def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y


def accuracy(preds, labels):
    correct_prediction = tf.equal(preds, labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return acc


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print 'losses:', losses
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd, is_train=True):
    """Helper to create an initialized Variable with weight decay.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    # var = _variable_on_cpu(name, shape,
    # tf.contrib.layers.xavier_initializer())

    if is_train:
        if FLAGS.convfc_initialize == 'xavier':
            print 'initialize', name, 'with xavier'
            init = tf.contrib.layers.xavier_initializer()
        elif FLAGS.convfc_initialize == 'orthogonal':
            print 'initialize', name, 'with orthogonal_initializer'
            init = orthogonal_initializer()
    else:
        print 'test phase, using constant_initializer(0.0)'
        init = tf.constant_initializer(0.0)

    assert init is not None, 'no initialzier for conv and fc set'

    var = _variable_on_cpu(name, shape, init)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def orthogonal_initializer(scale=1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    print('Warning -- You have opted to use the orthogonal_initializer function')

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  # this needs to be corrected to float32
        print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
