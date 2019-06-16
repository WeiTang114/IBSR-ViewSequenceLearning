
DATASET = 'my'
#DATASET = 'shape2vec'

if DATASET == 'my':
    SHAPE_LOL_TRAIN = './data/view/train_lists.txt'
    SHAPE_LOL_VAL = './data/view/val_lists.txt'
    SHAPE_LOL_TEST = './data/view/test_lists.txt'
    SHAPE_LOL_ALL = None
    IMAGE_LIST_TRAIN = './data/image/train.txt'
    IMAGE_LIST_VAL = './data/image/val.txt'
    IMAGE_LIST_TEST = './data/image/test.txt'

    SHAPE_LIST_PREFIX = ''
    SHAPE_VIEW_PREFIX = ''
    IMAGE_PREFIX = ''

    N_CLASSES = 40

elif DATASET == 'shape2vec':

    SHAPE_LOL_TRAIN = './data/shape/train_lists.txt'
    SHAPE_LOL_VAL = './data/shape/val_lists.txt'
    SHAPE_LOL_TEST = './data/shape/test_lists.txt'
    SHAPE_LOL_ALL = './data/shape/all_lists.txt'
    IMAGE_LIST_TRAIN = './data/image/train.txt'
    IMAGE_LIST_VAL = './data/image/val.txt'
    IMAGE_LIST_TEST = './data/image/test.txt'

    SHAPE_LIST_PREFIX = './data/'
    SHAPE_VIEW_PREFIX = './data/'
    IMAGE_PREFIX = './data/'

    N_CLASSES = 141


INPUT_W = 227
INPUT_H = 227

V = 12
# adjust N_POS_SAMPLE, N_NEG_SAMPLE and N_BATCH_IMAGES if OOM
N_POS_SAMPLE = 3
N_NEG_SAMPLE = 3
N_BATCH_IMAGES = 20

N_SHAPE_PER_CLASS = 3
BATCH_SIZE = 16

VAL_SAMPLE_SIZE = 256

# FEATURE_LAYER = 'conv4' # 'fc6' 'fc7
FEATURE_LAYER = 'pool5'  # 'fc6' 'fc7
# FEATURE_LAYER = 'conv5' # 'fc6' 'fc7
# FEATURE_LAYER = 'fc6' # 'fc6' 'fc7'

# "margin" for contrastive loss and triplet loss
CONTRASTIVE_LOSS_MARGIN = 0.2

# l2 normalize output feature
L2_NORMALIZATION = True

BATCH_NORM = False
# BATCH_NORM = False
BN_AFTER_ACTV = True  # conv -> relu -> bn
# BN_AFTER_ACTV = False  # conv -> bn -> relu


# for winston's sigmoid idea, replacing relu at conv5,
#  to suppress conv5 magnitude to 0~1
CONV5_SIGMOID = False


# external fc to reduce output dim
TAIL_LAYER = False
# TAIL_LAYER = True
TAIL_LAYER_DIM = 256

# GREY SCALE
IMAGE_GREYSCALE = True

# MVCNN shared weights
MVCNN_SHARED = True

# Cross domain adaptation experiments
#  both stream shared
CROSS_DOMAIN_SHARED = True
#  adaptation layer
CROSS_DOMAIN_ADAPTATION = True

# [conv4, pool5] feature concate, trying to combine appearance and semantic similarity
CONV4_POOL5_CONCAT = False

# initializer of conv and fc weights
# CONVFC_INIT = 'xavier'
CONVFC_INIT = 'orthogonal'

# 3d convolution for fusing
FUSE_3DCONV = True

print {k: v for k, v in locals().iteritems() if '__' not in k}
