#!/bin/sh

REPO_ROOT="$(cd ../../.. && pwd)"

python train.py \
    --data_path "$REPO_ROOT/Data/small_meshes" \
    --op_cache_dir "$REPO_ROOT/DiffusionML/op_cache" \
    --epochs 300 \
    --n_folds 1 \
    --C_latent 32 \
    --C_width 64 \
    --C_fate 0 \
    --dec_width 128 \
    --dec_layers 4 \
    --sphere_subdiv 3 \
    --beta_kl 1e-4 \
    --beta_warmup 100 \
    --lr 1e-3 \
    --lr_min 1e-5 \
    --lr_scheduler cosine \
    --weight_decay 1e-6 \
    --output_dir outputs/full_dataset \
    --sphere_cd_threshold 0.06
