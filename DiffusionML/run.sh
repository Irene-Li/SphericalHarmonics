#!/bin/sh

python DiffusionML/experiments/fate2hks/train.py \
    --data_path Data/20260211 \
    --epochs 10 \
    --n_folds 0 \
    --quality_percentile 100 \
    --C_width 32 \
    --lr 1e-2 \
    --output_dir DiffusionML/experiments/fate2hks/outputs \
