#!/bin/bash
# Usage:
# - modify $TMP value
# - mkdir $TMP
# - run command with evironment variables
#   $ PHASE=val CKPTITER=15000 GPU=3 ./run_eval_exp.sh


PHASE=${PHASE:-val}
#PHASE=test
#CKPTITER=21000
CKPTITER="${CKPTITER:-defaultvalue}"
GPU="${GPU:-0}"
TMP=./tmp/tmp

CVD="CUDA_VISIBLE_DEVICES=${GPU}"
CKPT=$TMP/model.ckpt-$CKPTITER
LAYER=pool5
FEATURE=$TMP/feature/
I_FEATURE=$FEATURE/image.$PHASE.$LAYER.$CKPTITER.hkl
S_FEATURE=$FEATURE/shape.$PHASE.$LAYER.$CKPTITER.hkl
RESULT=$TMP/result/result.$PHASE.$LAYER.$CKPTITER.hkl

# mv shared, cd shared, cd adapt experiments
MV_SHARED="--mvcnn_shared=True"
#MV_SHARED=""
CD_SHARED="--cd_shared=True"
#CD_SHARED=""
CD_ADAPT="--cd_adaptation=True"
#CD_ADAPT=""
FUSE_3DCONV="--fuse_3dconv=True"


mkdir -p $FEATURE
mkdir -p $TMP/result

CUDA_VISIBLE_DEVICES=${GPU} ipython extract_feature.py $PHASE $CKPT $LAYER $TMP/feature -- --convfc_initialize=xavier \
    $FUSE_3DCONV \
    $MV_SHARED \
    $CD_SHARED \
    $CD_ADAPT

CUDA_VISIBLE_DEVICES=${GPU} python retrieve.py $I_FEATURE $S_FEATURE $RESULT
python evaluation.py $RESULT 

echo "Gen RetrieVis"
python gen_retrievis.py $RESULT $TMP/visresult data/image/"$PHASE".txt data/view/"$PHASE"_lists.txt

echo ""
echo "Done: $RESULT"
echo ""
