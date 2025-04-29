# SatCLE
Official Implementation of WWW2025 "Nature Makes No Leaps: Building Continuous Location Embeddings with Satellite Imagery from the Web".

## Data
First download S2 dataset from https://github.com/microsoft/satclip and put it in ./data. \
For downstream dataset, please refer to the dataset in our paper, download it and put it in ./data/downstream.

## Environment

torch==1.13.1 \
pytorch-lightning==2.2.5 \
numpy==1.26.4


## Pretrain
bash train.sh

## Downstream Task
predict_*_satcle.ipynb