# gen resultX.txt files for RetrieVis
# input: result.npy: (M, N)
#        M queries, N databases

import numpy as np
import sys
import hickle as hkl
import os.path as p
import os
import errno


def main():
    argv = sys.argv
    assert len(argv) == 5, 'Usage: %s result_hkl vis_result_dir image_list.txt shape_lists.txt' % argv[0]

    result_hkl = argv[1]
    vis_result_dir = argv[2]
    image_list = argv[3]
    shape_lol = argv[4]

    # output dir prepare
    name = p.splitext(p.basename(result_hkl))[0]
    print name
    output_dir = p.join(vis_result_dir, name)
    mkdir_p(output_dir)

    # read image list
    image_paths = np.loadtxt(image_list, dtype=str)[:,0].tolist()

    # read shape view list 
    use_view = 2
    shape_list_paths = np.loadtxt(shape_lol, dtype=str)[:,0].tolist()
    view_paths = [np.loadtxt(l, dtype=str)[2+use_view] for l in shape_list_paths] 

    # load result data
    data = hkl.load(result_hkl) 
    q_label = data['query_label']
    db_label = data['db_label']
    result = data['result']

    for i, results in enumerate(result):
        clz = q_label[i]
        with open(output_dir + '/result%d_class%d.txt' % (i,clz), 'w+') as f:

            print>>f, image_paths[i]
            # print>>f, DB_DIR + '%d.0.png' % i)
            for j,resultid in enumerate(results):
                if j > 600:
                    break
                # print>>f, DB_DIR + '%d.%d.png' % (resultid / 12, resultid % 12)
                print>>f, view_paths[resultid] 
                # print>>f, DB_DIR + '%d.JPEG' % (resultid)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    main()

