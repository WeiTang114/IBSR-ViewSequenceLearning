import numpy as np
import sys
import cv2 # need to import before tf, issue:https://github.com/tensorflow/tensorflow/issues/1541
import tensorflow as tf
import hickle as hkl
print sys.argv

from config_loader import globals as g_
from input import ImageDataset, ShapeDataset
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")
tf.app.flags.DEFINE_string('weights', '', 
                            """finetune with a pretrained model""")

def extract_features(sess, dataset, input_ph, keep_prob_ph, phase_train_ph, feature_op):
    features = None
    batch_size = FLAGS.batch_size 
    print 'dataset size:', dataset.size()
    for i,(batch_x, _) in enumerate(dataset.batches(batch_size=batch_size)):
        print 'done ', i * batch_size
        feed = {input_ph: batch_x, keep_prob_ph: 1.0, phase_train_ph: False}
        feature = sess.run(feature_op, feed_dict=feed)
        
        # first iter
        if features is None:
            featuresize = feature.shape[1]
            features = np.zeros((dataset.size(), featuresize))
        
        features[i*batch_size : (i+1)*batch_size, ...] = feature
    
    return features

def outputhkl(filename, feature, label):
    hkl.dump({
        'feature': feature, 
        'label': label
    }, filename)
    


def main(phase, ckptfile, layer_name, outdir):
    image_lists = {
            'train': g_.IMAGE_LIST_TRAIN,
            'val': g_.IMAGE_LIST_VAL,
            'test': g_.IMAGE_LIST_TEST,
            'testall': g_.IMAGE_LIST_TEST
    }
    shape_lists = {
            'train': g_.SHAPE_LOL_TRAIN,
            'val': g_.SHAPE_LOL_VAL,
            'test': g_.SHAPE_LOL_TEST,
            'testall': g_.SHAPE_LOL_ALL
    }

    image_dataset = ImageDataset(image_lists[phase], subtract_mean=True, greyscale=g_.IMAGE_GREYSCALE, is_train=False, prefix=g_.IMAGE_PREFIX)
    shape_dataset = ShapeDataset(shape_lists[phase], subtract_mean=False, V=g_.V, prefix=g_.SHAPE_LIST_PREFIX, view_prefix=g_.SHAPE_VIEW_PREFIX)

    FLAGS.batch_size = 128
    
    with tf.Graph().as_default():
        
        image_ = tf.placeholder('float32', shape=(None, g_.INPUT_H, g_.INPUT_W, 3), name='image')
        view_ = tf.placeholder('float32', shape=(None, g_.V, g_.INPUT_H, g_.INPUT_W, 3), name='view')
        keep_prob_ = tf.placeholder('float32')
        phase_train_ = tf.placeholder(tf.bool, name='phase_train')
        image_feature = model.forward_image(image_, keep_prob_, layer_name, phase_train_)
        shape_feature = model.forward_shape(view_, keep_prob_, layer_name, phase_train_)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
        # restore with checkpoint file
        saver.restore(sess, ckptfile)
        print 'restore variables done'

        # restore with caffemodel
        # caffemodel = './alexnet_imagenet.npy'
        # sess.run(init_op)
        # model.load_alexnet_to_icnn(sess, caffemodel)
        # model.load_alexnet_to_mvcnn(sess, caffemodel)
        # print 'loading pretrained params done'


        image_features = extract_features(sess, image_dataset, image_, keep_prob_, phase_train_, image_feature)
        shape_features = extract_features(sess, shape_dataset, view_, keep_prob_, phase_train_, shape_feature)
    
    step = int(ckptfile.split('-')[-1])

    image_feature_output = '%s/image.%s.%s.%d.hkl' % (outdir, phase, layer_name, step)
    outputhkl(image_feature_output, image_features, np.array(image_dataset.labels))

    shape_feature_output = '%s/shape.%s.%s.%d.hkl' % (outdir, phase, layer_name, step)
    outputhkl(shape_feature_output, shape_features, np.array(shape_dataset.labels))
    

if __name__ == '__main__':
    # assert len(sys.argv) == 5, 'Usage: %s <train/val/test> ckptfile <conv4/pool5/fc6/fc7> outdir' % sys.argv[0] 

    phase = sys.argv[1]
    ckptfile = sys.argv[2]
    layer_name = sys.argv[3]
    outdir = sys.argv[4]
    main(phase, ckptfile, layer_name, outdir)
