#!/bin/sh
# Run from this directory: cd DiffusionML/experiments/hks_fate_coverage && sh run.sh
#
# Primary run: classify fate presence/absence (yes/no) for architecture validation.
#
# Loss: BCEWithLogitsLoss on raw logits (binary classification).
# No target normalisation needed — labels are already {0, 1}.
#
# Key settings:
#   - lr=1e-3: DiffusionNet collapses to majority-class prediction with lr=1e-4
#   - grad_clip=1.0: prevents instability in early training
#   - lr_scheduler=cosine: decays lr to lr_min over training run
#
# Expected behaviour:
#   - BCE loss should drop below the majority-class baseline
#   - Accuracy should exceed the majority-class baseline per fate

REPO_ROOT="$(cd ../../.. && pwd)"

python train.py \
    --data_path "$REPO_ROOT/Data/small_meshes" \
    --op_cache_dir "$REPO_ROOT/DiffusionML/op_cache" \
    --fate_names lgr sero lyz \
    --epochs 50 \
    --n_folds 1 \
    --C_width 64 \
    --N_block 4 \
    --mlp_hidden 32 \
    --lr 1e-3 \
    --weight_decay 1e-6 \
    --grad_clip 1.0 \
    --lr_scheduler cosine \
    --lr_min 1e-4 \
    --output_dir "$REPO_ROOT/DiffusionML/experiments/hks_fate_coverage/outputs"
