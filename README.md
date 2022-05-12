# DELG-pytorch
Pytorch Implementation of Unifying Deep Local and Global Features for Image Search ([delg-eccv20](https://arxiv.org/pdf/2001.05027.pdf))

- DELG pipline:
<p align="center"><img width="90%" src="tools/vis/delg_pipline.png" /></p>

## Installation

Install Python dependencies:

```
pip install -r requirements.txt
```

Set PYTHONPATH:

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Set Visible device

```
export CUDA_VISIBLE_DEVICES=1,2,...
```

## Training

Training a delg model:

```
python train_delg.py \
    --cfg configs/swin_tiny_delg_2gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.WEIGHTS path/to/pretrainedmodel

python train_delg.py \
    --cfg configs/config.yaml \
    OUT_DIR /Data_HDD/staff_xi_yang/LYMDataset/output2 \
    PORT 12312 \
    TRAIN.WEIGHTS /Data_HDD/yiming/DELG_clean_512size/checkpoints/model_epoch_0001.pyth
```

Resume training: 

```
python train_delg.py \
    --cfg configs/config.yaml \
    OUT_DIR /Data_HDD/staff_xi_yang/LYMDataset/output \
    PORT 12001 \
    TRAIN.AUTO_RESUME True
```

## Feature extraction

Extracting global and local feature for multi-scales
```
python tools/extractor_global.py --cfg configs/config.yaml
```
Refer [`extractor.sh`](tools/extract.sh) for using multicards

See [`visualize.ipynb`](tools/vis/attention/visualize.ipynb) for verification of local features

## Evaluation on ROxf and RPar

### Local Match

- Spatial Verification

    Install [**pydegensac**](https://github.com/ducha-aiki/pydegensac) and see **tools/rerank/spatial_verification.py**

- Examples
<p align="center"><img width="90%" src="tools/vis/matches/match_example_1.jpg" /></p>

- ASMK
    
    (https://github.com/jenicek/asmk)

### Results 

See (https://github.com/filipradenovic/revisitop) for details

```
cd tools/revisitop
python example_evaluate_with_local.py main
```

cd tools/revisitop
python example_evaluate.py main

- on roxford5k

|  Backbone | Train Size | Method | mAP E | mAP M | mAP H |
|--------------|:-------:|:------:|:-------:|:------------:|:-------------:|
|  ResNet50  |    224  |  Global Ranking                | 77.73 | **66.06** | 38.37 |
|  ResNet50  |    224  |  Global                        | 81.03 | **68.31** | 39.98 |
|  ResNet50  |    224  |  Global + Spatial Verification | 84.81 | **71.97** | 46.63 |
|  ResNet50  |    512  |  Global                        | 90.55 | **78.51** | 56.90 |
|  ResNet50  |    512  |  Global + Spatial Verification | 90.86 | **80.08** | 58.42 |

- on rparis6k(updating)

1. SOTA of R50-DELG is 78.3 mAP@M in the paper, we outperform it
2. All training set version is GLDv2-clean (81313, 1580470)
3. Traing size, global and local feature scales adopted are same with the paper


Test on pytorch 1.7.1 with py37