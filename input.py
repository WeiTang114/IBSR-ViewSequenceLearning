import cv2
import random
import numpy as np
import time
import multiprocessing as mp
import threading
import Queue
import fmq
from itertools import izip
from config_loader import globals as g_
from concurrent.futures import ThreadPoolExecutor

W = H = 256
OUT_W = g_.INPUT_W
OUT_H = g_.INPUT_H
QUEUE_FINISHED = '__QUEUE_FINISHED_13432ewrwe___' # random string

class Shape:
    def __init__(self, list_file, prefix):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]
        
        self.views = self._load_views(view_files, self.V, prefix)
        self.done_mean = False
        

    def _load_views(self, view_files, V, prefix):
        views = []
        for f in view_files:
            im = cv2.imread(prefix + f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            assert im.shape == (H, W, 3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)
        return views
    
    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.views[:,:,:,i] -= mean_bgr[i]
            
            self.done_mean = True
    
    def crop_center(self, size=(OUT_H, OUT_W)):
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        left = w / 2 - wn / 2
        top = h / 2 - hn / 2
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]
    
class Image:
    def __init__(self, path, label):
        with open(path) as f:
            self.label = label
        
        self.data = self._load(path)
        self.done_mean = False
        self.normalized = False

    def _load(self, path):
        im = cv2.imread(path)
        im = cv2.resize(im, (W, H))
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
        assert im.shape == (H, W, 3), 'BGR!'
        im = im.astype('float32')

        return im 
    
    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.data[:,:,i] -= mean_bgr[i]
            
            self.done_mean = True

    def normalize(self):
        if not self.normalized:
            self.data /= 256.
            self.normalized = True
    
    def crop_center(self, size=(OUT_H, OUT_W)):
        w, h = self.data.shape[0], self.data.shape[1]
        wn, hn = size
        left = w / 2 - wn / 2
        top = h / 2 - hn / 2
        self._crop(left, top, wn, hn)

    def random_crop(self, size=(OUT_H, OUT_W)):
        w, h = self.data.shape[0], self.data.shape[1]
        wn, hn = size
        left = random.randint(0, max(w - wn - 1, 0))
        top = random.randint(0, max(h - hn - 1, 0))
        self._crop(left, top, wn, hn)

    def _crop(self, left, top, w, h):
        right = left + w
        bottom = top + h
        self.data = self.data[left:right, top:bottom, :]

    def random_flip(self):
        if random.randint(0,1) == 1:
            self.data = self.data[:, ::-1, :]
            
    def to_greyscale(self):
        tmp = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        self.data = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)


class ShapeDataset:
    def __init__(self, list_of_lists_file, subtract_mean, V, prefix='', view_prefix=''):
        self.listfiles, self.labels = self._read_lists(list_of_lists_file, prefix)
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V
        self.view_prefix = view_prefix
        print 'dataset inited'
        print '  total size:', len(self.listfiles)
    
    def __getitem__(self, key):
        return self.listfiles[key], self.labels[key]

    def _read_lists(self, list_of_lists_file, prefix):
        print list_of_lists_file
        listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
        listfiles, labels  = zip(*[(prefix + l[0], int(l[1])) for l in listfile_labels])
        return listfiles, labels

    def load_shape(self, ind):
        s = Shape(self.listfiles[ind], self.view_prefix)
        s.crop_center()
        if self.subtract_mean:
            s.subtract_mean()
        return s 

    def sample_from_class(self, clz, n):
        inds_of_clz = [i for i,lab in enumerate(self.labels) if lab == clz]
        # print 'sample_from_class', clz, inds_of_clz
        return self._sample(inds_of_clz, n)

    def sample_not_class(self, clz, n):
        inds_not_clz = [i for i,lab in enumerate(self.labels) if lab != clz]
        # print 'sample_not_class', clz, inds_not_clz
        return self._sample(inds_not_clz, n)

    def _sample(self, items, n):
        """
        ignore size limit (len(items) < n)
        """
        try:
            return random.sample(items, n)
        except ValueError:
            results = []
            for i in range(n):
                results.append(random.choice(items))
            return results

    def batches(self, batch_size):
        for x,y in self._batches_fast(self.listfiles, batch_size):
            yield x,y
        
    def _batches_fast(self, listfiles, batch_size):
        n = len(listfiles)

        def load(inds, q):                    
            for ind in inds:
                q.put(self.load_shape(ind))

            # indicate that I'm done
            q.put(None)
            # q.close()

        q = fmq.Queue(maxsize=256)

        # background loading Shapes process
        p = mp.Process(target=load, args=(range(len(listfiles)), q))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()

        x = np.zeros((batch_size, self.V, OUT_H, OUT_W, 3)) 
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()
            
            # print 'q size', q.qsize() 

            for j in xrange(batch_size):
                s = q.get()

                # queue is done
                if s is None: 
                    x = np.delete(x, range(j, batch_size), axis=0)
                    y = np.delete(y, range(j, batch_size), axis=0)
                    break
                
                x[j, ...] = s.views
                y[j] = s.label 
            
            # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y


    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.listfiles)


class ImageDataset:
    def __init__(self, imagelist_file, subtract_mean, greyscale, is_train, prefix=''):
        self.image_paths, self.labels = self._read_imagelist(imagelist_file, prefix)
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.greyscale = greyscale
        self.is_train = is_train
        print 'image dataset inited'
        print '  total size:', len(self.image_paths)


    def __getitem__(self, key):
        return self.image_paths[key], self.labels[key]

    def _read_imagelist(self, listfile, prefix):
        path_and_labels = np.loadtxt(listfile, dtype=str).tolist()
        paths, labels = zip(*[(prefix + l[0], int(l[1])) for l in path_and_labels])
        return paths, labels

    def shuffle(self):
        inds = range(len(self.image_paths))

        random.shuffle(inds)
        self.image_paths = [self.image_paths[i] for i in inds]
        self.labels = [self.labels[i] for i in inds]
        self.shuffled = True

    def load_image(self, ind, augmentation=False):
        path, label = self[ind]
        i = Image(path, label)       

        if not augmentation:
            i.crop_center()
        else:
            i.random_crop()
            i.random_flip()

        if self.subtract_mean:
            i.subtract_mean()

        if self.greyscale:
            i.to_greyscale()

        return i

    def label(self, ind):
        return self.labels[ind]
    
    def batches(self, batch_size):
        for x,y in self._batches_fast(batch_size):
            yield x,y
        
    def _batches_fast(self, batch_size):
        paths, labels = self.image_paths, self.labels
        n = len(paths)

        def load(inds, q):                    
            for ind in inds:
                q.put(self.load_image(ind, self.is_train))

            # indicate that I'm done
            q.put(None)
            # q.close()

        q = fmq.Queue(maxsize=256)

        # background loading Shapes process
        p = mp.Process(target=load, args=(range(len(paths)), q))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()

        x = np.zeros((batch_size, OUT_H, OUT_W, 3)) 
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()
            
            # print 'q size', q.qsize() 

            for j in xrange(batch_size):
                image = q.get()

                # queue is done
                if image is None: 
                    x = np.delete(x, range(j, batch_size), axis=0)
                    y = np.delete(y, range(j, batch_size), axis=0)
                    break
                
                x[j, ...] = image.data
                y[j] = labels[i + j] 
            
            # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.image_paths)


class Dataset:
    def __init__(self, image_dataset, shape_dataset, name, is_train):
        self.imageset = image_dataset
        self.shapeset = shape_dataset
        self.name = name
        self.triplets = [] # [((i_id, v_id), y), ..], y = 0->match, 1->mismatch
        self.shuffled = False
        self.is_train = is_train
        print 'dataset inited'

    def sample_triplets(self, pos_n, neg_n):
        assert pos_n == neg_n, 'pos_n != neg_n for triplets'
        starttime = time.time()

        triplets = []
        for i, (_, clz) in enumerate(self.imageset):
            pos_shape_inds = self.shapeset.sample_from_class(clz, pos_n)
            neg_shape_inds = self.shapeset.sample_not_class(clz, neg_n)
            triplets.extend([(i, j, k) for j,k in zip(pos_shape_inds, neg_shape_inds)])

        self.triplets = triplets

        print '(%s) sample_triplets time: %f sec' % (self.name, time.time() - starttime)

    
    def _sampled_triplets_or_fail(self):
        assert len(self.triplets) > 0, '(%s) sample_triplets() need to be called first' % self.name
    
    
    def shuffle(self):
        self._sampled_triplets_or_fail()
        random.shuffle(self.triplets)
        self.shuffled = True


    def batches(self, batch_size):
        self._sampled_triplets_or_fail()
        for x in self._batches_fast(self.triplets, batch_size):
            yield x
        
    def sample_batches(self, batch_size, n):
        self._sampled_triplets_or_fail()
        triplets = random.sample(self.triplets, n)
        for x in self._batches_fast(triplets, batch_size):
            yield x


    def _batches_fast(self, triplets, batch_size):

        def load_images(inds, q):                    
            for ind in inds:
                q.put(self.imageset.load_image(ind, augmentation=self.is_train))

            # indicate that I'm done
            q.put(QUEUE_FINISHED)
            # q.close()
        
        def load_shapes(inds, q):
            for ind in inds:
                q.put(self.shapeset.load_shape(ind))
            
            # indicate that I'm done
            q.put(QUEUE_FINISHED)
            # q.close()


        image_inds = [triplet[0] for triplet in triplets]
        shape_pos_inds = [triplet[1] for triplet in triplets]
        shape_neg_inds = [triplet[2] for triplet in triplets]

        q_image = fmq.Queue(maxsize=128)
        q_shape_pos = fmq.Queue(maxsize=128)
        q_shape_neg = fmq.Queue(maxsize=128)

        # background loading Shapes process
        p1 = mp.Process(target=load_images, args=(image_inds, q_image))
        p2 = mp.Process(target=load_shapes, args=(shape_pos_inds, q_shape_pos)) 
        p3 = mp.Process(target=load_shapes, args=(shape_neg_inds, q_shape_neg)) 

        # daemon child is killed when parent exits
        p1.daemon = True
        p2.daemon = True
        p3.daemon = True
        p1.start()
        p2.start()
        p3.start()

        x_image = np.zeros((batch_size, OUT_H, OUT_W, 3)) 
        x_views_pos = np.zeros((batch_size, self.shapeset.V, OUT_H, OUT_W, 3))
        x_views_neg = np.zeros((batch_size, self.shapeset.V, OUT_H, OUT_W, 3))

        for i in xrange(0, len(triplets), batch_size):
            starttime = time.time()
            
            # print 'q size', q.qsize() 

            for j in xrange(batch_size):
                image = q_image.get()
                shape_pos = q_shape_pos.get()
                shape_neg = q_shape_neg.get()

                # queue is done
                if image is QUEUE_FINISHED or shape_pos is QUEUE_FINISHED or shape_neg is QUEUE_FINISHED:
                    x_image = np.delete(x_image, range(j, batch_size), axis=0)
                    x_views_pos = np.delete(x_views_pos, range(j, batch_size), axis=0)
                    x_views_neg = np.delete(x_views_neg, range(j, batch_size), axis=0)
                    break
                
                x_image[j, ...] = image.data
                x_views_pos[j, ...] = shape_pos.views
                x_views_neg[j, ...] = shape_neg.views
            
            # print 'load batch time:', time.time()-starttime, 'sec'
            yield (x_image, x_views_pos, x_views_neg)

    def size(self):
        self._sampled_triplets_or_fail()
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.triplets)



class TripletSamplingDataset:
    def __init__(self, image_dataset, shape_dataset, name, is_train):
        self.imageset = image_dataset
        self.shapeset = shape_dataset
        self.name = name
        self.image_inds = range(image_dataset.size())
        self.shuffled = False
        self.is_train = is_train
        self.queue = None
        print 'dataset inited'

    def shuffle(self):

        inds = range(self.imageset.size())
        self.image_inds = inds
        random.shuffle(self.image_inds)
        self.shuffled = True


    def batches(self, n_image, n_shape_per_class, n_class):
        for x in self._batches_fast(n_image, n_shape_per_class, n_class):
            yield x

    def _batches_fast(self, n_image, n_shape_per_class, n_class):
        image_inds = self.image_inds

        q = Queue.Queue(maxsize=16) # 32 batches = a lot of images + shapes
        self.queue = q

        # background loading Shapes process
        t = threading.Thread(target=self.load_batches, 
                args=(image_inds, q, n_image, n_shape_per_class, n_class))

        # daemon child is killed when parent exits
        t.daemon = True
        t.start()

        for i in xrange(0, len(image_inds), n_image):
            starttime = time.time()
            
            # print 'q size', q.qsize() 
            item = q.get()
            if item is QUEUE_FINISHED:
                break

            try: 
                x_images, y_images, x_views, y_views = item
            except ValueError as e:
                print 'ValueError item ????'
                print item
                print str(e)
                break

            # print 'load batch time:', time.time()-starttime, 'sec'
            yield (x_images, y_images, x_views, y_views)


    def load_batches(self, image_inds, q, n_image, n_shape_per_class, n_class):
        for i_inds in grouped(image_inds, n_image):
            # load batch image
            images = []
            image_labels = []
            for ind in i_inds:
                image = self.imageset.load_image(ind, augmentation=self.is_train)
                images.append(image.data)
                image_labels.append(image.label)
            images = np.array(images)

            # sample batch shape inds
            s_inds = []
            for clz in xrange(n_class):
                s_inds.extend(self.shapeset.sample_from_class(clz, n_shape_per_class))
            
            def load_shape(ind):
                shape = self.shapeset.load_shape(ind)
                return shape.views, shape.label

            s_views = []
            s_labels = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for (views, label) in pool.map(load_shape, s_inds):
                    s_views.append(views)
                    s_labels.append(label)
            s_views = np.array(s_views)

            batch = (images, image_labels, s_views, s_labels)
            q.put(batch)

        # indicate that I'm done
        q.put(QUEUE_FINISHED)

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.image_inds)

    def print_status(self):
        msg = 'TripletSamplingDataset: Queue: [ %d / %d ]' % (self.queue.qsize(), self.queue.maxsize)
        print msg


class TripletSamplingDatasetS2V(TripletSamplingDataset):
    
    def shuffle(self):

        inds = range(self.imageset.size())

        """ad-hoc tuning for SHAPE2VEC TRAINING"""
        n_im_class = 64
        n_class = 141
        for i in range(n_class):
            part = inds[i * n_im_class: (i+1) * n_im_class]
            random.shuffle(part)
            inds[i * n_im_class: (i+1) * n_im_class] = part
        old_inds = list(inds)
        inds = []
        for i in range(n_class):
            part = old_inds[i::n_im_class]
            random.shuffle(part)
            inds.extend(part)
        """"""

        self.image_inds = inds
        self.shuffled = True


    def load_batches(self, image_inds, q, n_image, n_shape_per_class, n_class):
        for i_inds in grouped(image_inds, n_image):
            # load batch image
            images = []
            image_labels = []
            for ind in i_inds:
                image = self.imageset.load_image(ind, augmentation=self.is_train)
                images.append(image.data)
                image_labels.append(image.label)
            images = np.array(images)
            # print image_labels

            # sample batch shape inds
            s_inds = []
            for clz in image_labels:
                s_inds.extend(self.shapeset.sample_from_class(clz, n_shape_per_class))

            # Trying to make process pool for loading shapes but failed
            def load_shape(ind):
                shape = self.shapeset.load_shape(ind)
                return shape.views, shape.label
                
            s_views = []
            s_labels = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for (views, label) in pool.map(load_shape, s_inds):
                    s_views.append(views)
                    s_labels.append(label)
            s_views = np.array(s_views)

            batch = (images, image_labels, s_views, s_labels)
            q.put(batch)


        # indicate that I'm done
        q.put(QUEUE_FINISHED)
        # q.close()


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return izip(*[iter(iterable)]*n)

