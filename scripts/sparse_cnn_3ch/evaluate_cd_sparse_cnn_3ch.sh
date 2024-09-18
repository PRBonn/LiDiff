#!/bin/bash

cd /home/ekirby/workspace/LiDiff

pip install -e .

cd lidiff

python train_objects_full_diff.py -c config/sparse_cnn_ablation/scnn_multiclass_gen.yaml -w checkpoints/lidiff/checkpoints/all_objects_cleaned_aligned_1_epoch=499.ckpt --test
