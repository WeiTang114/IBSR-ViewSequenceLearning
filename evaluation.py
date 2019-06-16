import sys
import numpy as np
import hickle as hkl
import argparse
import math


def calc_map(result, labs_query, labs_db, ignorefirst=False):
    aps = []
    for i, db_inds in enumerate(result):
        lab_q  = labs_query[i]
        lab_db = labs_db[db_inds]
        if ignorefirst:
            lab_db = lab_db[1:]

        total_ap = 0.
        hit_cnt = 0. 
        precisions = []
        for j, l in enumerate(lab_db):
            if l == lab_q:
                hit_cnt += 1.
                total_ap += (hit_cnt / (j+1))
                precisions.append(hit_cnt / (j+1))
        total_ap /= hit_cnt
        aps.append(total_ap)
        
        # print total_ap
        print '%d class %d, first match: %f' % (i, lab_q, 1 / precisions[0])

    # mean ap
    m_ap = np.mean(aps)
    print 'map:', m_ap


    # per class
    classes = sorted(list(set(labs_query)))
    classes_aps = [[ap for (ap, clz) in zip(aps, labs_query) if clz == c] for c in classes]
    classes_aps = [np.mean(ap) for ap in classes_aps]
    for ap, clz in zip(classes_aps, classes):
        print 'class%d\t%f' % (clz, ap)

    return m_ap


def ndcg(result):
    ndcgs = []
    for r in result:
        x = ndcg_single(r)
        ndcgs.append(x)
    ndcg_mean = np.mean(ndcgs)
    print 'NDCG:', ndcg_mean 

    
def dcg(x):

    n = len(x)
    logs = np.log2(range(2, n + 2)) # 2, 3, 4, ..
    ws = (1.0 / logs) # 1, 1/log(3), ...
    gains = ws * x
    return np.sum(gains)

    # total = x[0]
    # for i, _ in enumerate(x[1:]):
        # w = 1.0 / logs[i]
        # total += x[i] * w
    # return total


def idcg(x):
    return dcg(sorted(x, reverse=True))


def ndcg_single(x):
    i = idcg(x)
    if i == 0.0:
        return 0.0
    return dcg(x) / i
    

def nearest_neighbor(result):
    hit = np.count_nonzero(result[:, 0])
    nn = hit / float(result.shape[0])
    print 'NN:', nn
    return nn

def tier(result, t):
    assert t == 1 or t == 2
    
    


def boolize_result(result, labs_query, labs_db):
    labs_db = np.array(labs_db)
    result2 = np.array(result)
    for i, (y1, res) in enumerate(zip(labs_query, result)):
        for j, y2 in enumerate(labs_db[res]):
            result2[i, j] = int(y1 == y2)
    return result2 

if __name__ == '__main__':
    argv = sys.argv
    
    parser = argparse.ArgumentParser(description='evaluate retrieval result, mAP')
    parser.add_argument('resultfile', help='result hkl file')
    parser.add_argument('--ignorefirst', action='store_true',
                        help='ignore the first matching. Usually because the query is in the database')
    args = parser.parse_args()
    
    
    data = hkl.load(args.resultfile)
    image_label = data['query_label']
    shape_label = data['db_label']
    result = data['result']

    print 'map for image to view'
    calc_map(result, image_label, shape_label, ignorefirst=args.ignorefirst)
    
    result_01 = boolize_result(result, image_label, shape_label)
    ndcg(result_01)
    nearest_neighbor(result_01)

