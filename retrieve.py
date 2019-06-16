import numpy as np
import multiprocessing as mp
import sys
from scipy.spatial.distance import cdist
import hickle as hkl
import tensorflow as tf

def main():
    assert len(sys.argv) == 4, 'Usage: %s hkl_image hkl_shape hkl_result' % sys.argv[0]

    hkl_image = sys.argv[1]
    hkl_shape = sys.argv[2]
    hkl_result = sys.argv[3]

    print 'feature img:', hkl_image
    print 'feature shape:', hkl_shape
    print 'result:', hkl_result

    image_data = hkl.load(hkl_image)
    i_feature, i_label = image_data['feature'], image_data['label']

    shape_data = hkl.load(hkl_shape)
    s_feature, s_label = shape_data['feature'], shape_data['label']

    print 'start cdist, feature dim: %d, query n: %d, db n: %d' % (i_feature.shape[1], i_feature.shape[0], s_feature.shape[0])
    dist = tf_cdist(i_feature, s_feature)
    print 'done cdist'

    print 'start argsort'
    result = np.argsort(dist, axis=1) 
    print 'done argsort'

    hkl.dump({
        'query_label': i_label,
        'db_label': s_label,
        'result': result
    }, hkl_result)


def tf_cdist(a, b):
    """cdist (squared euclidean) with tensorflow"""

    def distance_matrix(A, B):
        """A is a (N, Dim) matrix"""
        # http://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        r_A = tf.reshape(tf.reduce_sum(A * A, 1), [-1, 1])
        r_B = tf.reshape(tf.reduce_sum(B * B, 1), [-1, 1])
        D = r_A - 2 * tf.matmul(A, tf.transpose(B)) + tf.transpose(r_B)
        return D

    A = tf.constant(a)
    B = tf.constant(b)
    sess = tf.Session()
    return sess.run(distance_matrix(A, B))
    


def retrieve_cosinesim(hog_imgs, hog_views):
    retrieved_inds = []
    for i, hog_i in enumerate(hog_imgs):
        if i % 100 == 0: print i
        distance_views = 1. - (hog_i * hog_views).sum(axis=1) \
                / (0.00001 + np.linalg.norm(hog_i) * np.linalg.norm(hog_views, axis=1))
        distances = np.zeros(N_MODELS, dtype='float')
        # for j in xrange(N_MODELS):
            # distances[j] = distance_views[j*N_VIEWS: (j+1)*N_VIEWS].min()
            # distances += distance_views[j::N_VIEWS]
        # inds = np.argsort(distances)
        inds = np.argsort(distance_views)
        
        retrieved_inds.append(inds)

    result = np.array(retrieved_inds)
    # assert result.shape == (hog_i.shape[0], N_MODELS), 'shape:' + result.shape
    print result.shape
    return result


def cosinesim(args):
    (hog_i, hog_views) = args
    distance_views = 1. - (hog_i * hog_views).sum(axis=1) \
            / (0.00001 + np.linalg.norm(hog_i) * np.linalg.norm(hog_views, axis=1))
            
    distances = np.zeros(N_MODELS, dtype='float')
    # inds = np.argsort(distance_views)

    
    for j in xrange(N_MODELS):
        distances[j] = distance_views[j*N_VIEWS: (j+1)*N_VIEWS].min()
    inds = np.argsort(distances)

    return inds
        
def retrieve_cosinesim_mp(hog_imgs, hog_views):

    pool = mp.Pool(32)
    retrieved_inds = pool.map(cosinesim, zip(list(hog_imgs), [hog_views]*len(hog_imgs)))
    
    result = np.array(retrieved_inds)

    # assert result.shape == (hog_i.shape[0], N_MODELS), 'shape:' + result.shape
    print result.shape
    return result

def retrieve_l2_mp(hog_imgs, hog_views):

    print    hog_imgs.shape
    print hog_views.shape

    pool = mp.Pool(32)
    retrieved_inds = pool.map(l2distance, zip(list(hog_imgs), [hog_views]*len(hog_imgs)))
    
    result = np.array(retrieved_inds)

    # assert result.shape == (hog_i.shape[0], N_MODELS), 'shape:' + result.shape
    print result.shape
    return result

def l2distance(args):
    (query, db) = args
    distances_views = ((db - query) ** 2).sum(axis=1) ** 0.5
    distances = np.zeros(N_MODELS, dtype='float')
    for i in xrange(N_MODELS):
        distances[i] = np.min(distances_views[i*12: (i+1)*12])  
    
    inds = np.argsort(distances)

    return inds

def retrieve_l2_allviews(hog_imgs, hog_views):

    retrieved_inds = []
    for i, hog_i in enumerate(hog_imgs):
        if i % 100 == 0: print i
        # distance_views = np.abs(hog_views - hog_i).sum(axis=1)
        distance_views = ((hog_views - hog_i) ** 2).sum(axis = 1)**0.5
        distances = np.zeros(N_MODELS, dtype='float')
        # for j in xrange(N_MODELS):
            # distances[j] = distance_views[j*N_VIEWS: (j+1)*N_VIEWS].min()
            # distances += distance_views[j::N_VIEWS]
        # inds = np.argsort(distances)
        inds = np.argsort(distance_views)
        
        retrieved_inds.append(inds)

    result = np.array(retrieved_inds)
    # assert result.shape == (hog_i.shape[0], N_MODELS), 'shape:' + result.shape
    print result.shape
    return result
    
def retrieve_l2(hog_imgs, hog_views):

    retrieved_inds = []
    for i, hog_i in enumerate(hog_imgs):
        if i % 100 == 0: print i
        # distance_views = np.abs(hog_views - hog_i).sum(axis=1)
        distance_views = ((hog_views - hog_i) ** 2).sum(axis = 1)**0.5
        distances = np.zeros(N_MODELS, dtype='float')
        for j in xrange(N_MODELS):
            distances[j] = distance_views[j*N_VIEWS: (j+1)*N_VIEWS].min()
            # distances += distance_views[j::N_VIEWS]
        inds = np.argsort(distances)
        # inds = np.argsort(distance_views)
        
        retrieved_inds.append(inds)

    result = np.array(retrieved_inds)
    # assert result.shape == (hog_i.shape[0], N_MODELS), 'shape:' + result.shape
    print result.shape
    return result

if __name__ == '__main__':
    main()

