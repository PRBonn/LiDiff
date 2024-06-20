#!/bin/bash

cd /home/ekirby/workspace/LiDiff

# Directory containing the config files
config_dir="lidiff/config/object_generation"
weights_dir="lidiff/checkpoints"

config="$config_dir/config_bike_gen_aligned.yaml"
weights="$weights_dir/bike_gen_aligned_1_epoch=499.ckpt"

python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n bike_gen_aligned_1_val -t recreate -cls vehicle.bicycle -s val
