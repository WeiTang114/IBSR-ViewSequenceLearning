# Readme

The implementation of Cross-Domain Image-Based 3D Shape Retrieval by View Sequence Learning.

```
Lee, Tang, et al. "Cross-Domain Image-Based 3D Shape Retrieval by View Sequence Learning." 2018 International Conference on 3D Vision (3DV). IEEE, 2018.
```

## Usage

Files descriptions：

1. globals.py: experiment configs.
2. input.py: a small lib to load image / 3D shape files.
3. run_eval_exp.sh: script for running experiments.

### Prepare dataset

1. Download dataset [here](https://www.cmlab.csie.ntu.edu.tw/~weitang114/ibsr/IM2MN.zip).
2. unzip it into the repository directory.

```
unzip IM2MN.zip
```

### train

```bash
mkdir tmp
```

Prepare pretrained model:

```bash
./prepare_pretrained_alexnet.sh
```

train with pretrained model:

```bash
python train.py --caffemodel ./alexnet_imagenet.npy  --train_dir tmp --learning_rate 0.01
```

fine-tune with checkpoint file:
```bash
python train.py --weights tmp/model.ckpt-10000 --train_dir tmp --learning_rate 0.01
```

### test

```
PHASE=val CKPTITER=15000 GPU=1 ./run_eval_exp.sh
```

This scripts does 4 steps for testing:

1. extract features of images and 3D shapes. Save them into hkl files.
2. retrieve. The results are saved into hkl files.
3. evaluate. Print mAP.
4. gen_retrievis.py. Generate result files for [RetrieVis](https://github.com/WeiTang114/RetrieVis) visualization.

### visualize

```bash
git clone https://github.com/WeiTang114/RetrieVis
cd RetrieVis/
python simple.py <ibsr-triplet>/tmp/visresult/result.test.pool5.15000 --port <port>
```

Connect to http://\<ip\>:\<port\> to see the results。
