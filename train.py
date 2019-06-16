import numpy as np
import os
import sys
import inspect
import cv2  # need to import before tf, issue:https://github.com/tensorflow/tensorflow/issues/1541
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob
import random
from utils import Timer

from input import ImageDataset, ShapeDataset, Dataset, TripletSamplingDatasetS2V, TripletSamplingDataset


currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model
from config_loader import globals as g_

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp3/weitang114/MVCNN-TF/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                           """finetune with a pretrained model""")
tf.app.flags.DEFINE_integer('n_views', 12,
                            """Number of views rendered from a mesh.""")
tf.app.flags.DEFINE_string(
    'caffemodel',
    '',
    """finetune with a model converted by caffe-tensorflow""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.

np.set_printoptions(precision=3)


def train(dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print 'train() called'
    is_finetune = bool(ckptfile)
    V = FLAGS.n_views
    batch_size = FLAGS.batch_size
    mvcnn_shared = FLAGS.mvcnn_shared
    cd_shared = FLAGS.cd_shared

    data_size = dataset_train.imageset.size() * (g_.N_POS_SAMPLE + g_.N_NEG_SAMPLE)
    print 'training size:', data_size

    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        global_step = tf.Variable(startstep, trainable=False)

        image_ = tf.placeholder(
            'float32',
            shape=(
                None,
                g_.INPUT_H,
                g_.INPUT_W,
                3),
            name='image')
        view_ = tf.placeholder(
            'float32',
            shape=(
                None,
                V,
                g_.INPUT_H,
                g_.INPUT_W,
                3),
            name='view_pos')
        image_label_ = tf.placeholder('int32', shape=(None), name='image_label')
        keep_prob_ = tf.placeholder('float32', name='keep_prob')
        phase_train_ = tf.placeholder(tf.bool, name='phase_train')

        image_feature = model.forward_image(
            image_, keep_prob_, FLAGS.layer, phase_train_)
        shape_feature = model.forward_shape(
            view_, keep_prob_, FLAGS.layer, phase_train_)
        if g_.DATASET == 'my':
            loss = model.loss(
                image_feature,
                shape_feature,
                image_label_,
                g_.N_BATCH_IMAGES,
                g_.N_SHAPE_PER_CLASS,
                g_.N_CLASSES)
        elif g_.DATASET == 'shape2vec':
            loss = model.loss_shape2vec(
                image_feature,
                shape_feature,
                image_label_,
                g_.N_BATCH_IMAGES,
                g_.N_SHAPE_PER_CLASS,
                g_.N_BATCH_IMAGES)
        train_op = model.train(loss, global_step, data_size)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.summary.merge_all()

        # must be after merge_all_summaries
        validation_loss = tf.placeholder(
            'float32', shape=(), name='validation_loss')
        validation_summary = tf.summary.scalar(
            'validation_loss', validation_loss)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement,
                gpu_options=gpu_options))

        if is_finetune:
            saver.restore(sess, ckptfile)
            print 'restore variables done'
        elif caffemodel:
            sess.run(init_op)
            model.load_alexnet_to_icnn(sess, caffemodel)
            if mvcnn_shared:
                if not cd_shared:
                    model.load_alexnet_to_mvcnn_shared(sess, caffemodel)
            else:
                model.load_alexnet_to_mvcnn(sess, caffemodel)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            # from scratch
            sess.run(init_op)
            print 'init_op done'

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph)

        step = startstep

        n_classes = g_.N_CLASSES
        n_batch_images = g_.N_BATCH_IMAGES
        n_shape_per_class = g_.N_SHAPE_PER_CLASS
        n_triplets_per_batch = FLAGS.batch_size
        margin = FLAGS.margin

        imageset_train, shapeset_train = dataset_train.imageset, dataset_train.shapeset

        print 'start training'
        for epoch in xrange(400):
            # imageset_train.shuffle()
            dataset_train.shuffle()

            dataset_val.sample_triplets(g_.N_POS_SAMPLE, g_.N_NEG_SAMPLE)
            dataset_val.shuffle()

            # good image shape loading
            for batch in dataset_train.batches(
                    n_batch_images, n_shape_per_class, n_classes):
                start_time = time.time()
                step += 1
                ims, im_labels, shapes, sh_labels = batch

                # dataset_train.print_status()

                # extract image and shape features
                with Timer() as t:
                    feed = {image_: ims, keep_prob_: 1.0, phase_train_: False}
                    im_features = sess.run(image_feature, feed_dict=feed)
                # print 'image feature time:', t, 'sec'

                time1 = time.time()
                # print 'time1:', time1-start_time

                # feed into network
                feed = {image_: ims,
                        view_: shapes,
                        image_label_: im_labels,
                        keep_prob_: 0.5,
                        phase_train_: True}

                # run
                _, loss_value = sess.run([train_op, loss], feed_dict=feed)

                time5 = time.time()
                # print 'run train_op time:', time5 - time4, 'sec'

                duration = time.time() - start_time

                # assert not np.isnan(loss_value), 'Model diverged with loss = NN'
                if np.isnan(loss_value):
                    'Model diverged with loss = NN'

                if step % 10 == 0 or step - startstep < 30:
                    sec_per_batch = float(duration)
                    print '%s: step %d, loss=%.4f (%.1f examples/sec; %.3f sec/batch)' \
                        % (datetime.now(), step, loss_value,
                           FLAGS.batch_size/duration, sec_per_batch)

                # val
                if step % 100 == 0 or step == 1:  # and step > 0:
                    print 'doing validation'
                    val_start_time = time.time()
                    val_losses = []
                    predictions = np.array([])

                    val_y = []
                    for val_step, (val_batch_x_i, val_batch_x_v_pos, val_batch_x_v_neg) in \
                            enumerate(dataset_val.sample_batches(batch_size, g_.VAL_SAMPLE_SIZE)):
                        val_feed_dict_i = {
                            image_: val_batch_x_i,
                            keep_prob_: 1.0,
                            phase_train_: False}
                        val_feed_dict_v_pos = {
                            view_: val_batch_x_v_pos, keep_prob_: 1.0, phase_train_: False}
                        val_feed_dict_v_neg = {
                            view_: val_batch_x_v_neg, keep_prob_: 1.0, phase_train_: False}
                        val_feature_i = sess.run(
                            image_feature, feed_dict=val_feed_dict_i)
                        val_feature_v_pos = sess.run(
                            shape_feature, feed_dict=val_feed_dict_v_pos)
                        val_feature_v_neg = sess.run(
                            shape_feature, feed_dict=val_feed_dict_v_neg)
                        val_loss = model.test_loss_np(
                            val_feature_i, val_feature_v_pos, val_feature_v_neg)
                        # print val_batch_y[:20]
                        # print pred[:20]
                        val_losses.append(val_loss)

                    val_loss = np.mean(val_losses)
                    print '%s: step %d, validation loss=%.4f, takes %f sec' %\
                        (datetime.now(), step, val_loss, time.time() - val_start_time)

                    # validation summary
                    val_loss_summ = sess.run(
                        validation_summary, feed_dict={
                            validation_loss: val_loss})
                    summary_writer.add_summary(val_loss_summ, step)
                    summary_writer.flush()

                if step % 100 == 0 or step in [10, 20, 30]:
                    # print 'running fucking summary'
                    summary_str = sess.run(summary_op, feed_dict=feed)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step % 1000 == 0 or (step+1) == FLAGS.max_steps or (0 < step < 1000 and step % 200 == 0)\
                        and step > startstep:
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv):
    st = time.time()
    print 'start loading data'

    shape_dataset_train = ShapeDataset(
        g_.SHAPE_LOL_TRAIN,
        subtract_mean=False,
        V=12,
        prefix=g_.SHAPE_LIST_PREFIX,
        view_prefix=g_.SHAPE_VIEW_PREFIX)
    image_dataset_train = ImageDataset(
        g_.IMAGE_LIST_TRAIN,
        subtract_mean=True,
        is_train=True,
        greyscale=g_.IMAGE_GREYSCALE,
        prefix=g_.IMAGE_PREFIX)
    # dataset_train = Dataset(image_dataset_train, shape_dataset_train, name='train', is_train=True)
    if g_.DATASET == 'my':
        TSDataset = TripletSamplingDataset
    elif g_.DATASET == 'shape2vec':
        TSDataset = TripletSamplingDatasetS2V

    dataset_train = TSDataset(
        image_dataset_train,
        shape_dataset_train,
        name='train',
        is_train=True)

    shape_dataset_val = ShapeDataset(
        g_.SHAPE_LOL_VAL,
        subtract_mean=False,
        V=12,
        prefix=g_.SHAPE_LIST_PREFIX,
        view_prefix=g_.SHAPE_VIEW_PREFIX)
    image_dataset_val = ImageDataset(
        g_.IMAGE_LIST_VAL,
        subtract_mean=True,
        is_train=False,
        greyscale=g_.IMAGE_GREYSCALE,
        prefix=g_.IMAGE_PREFIX)
    dataset_val = Dataset(
        image_dataset_val,
        shape_dataset_val,
        name='val',
        is_train=False)

    print 'done loading data, time=', time.time() - st

    train(dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)


if __name__ == '__main__':
    main(sys.argv)
