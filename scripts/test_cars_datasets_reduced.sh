#!/bin/bash

cd /home/ekirby/workspace/LiDiff

# Directory containing the config files
config_dir="lidiff/config/object_generation"
weights_dir="lidiff/checkpoints"

config_cars_reduced="$config_dir/config_car_gen_reduced_dataset.yaml"
weights_cars_reduced="$weights_dir/car_gen_dataset_limited_1_epoch=99.ckpt"

echo "Running recreation for cars reduced dataset using condition from train"
python lidiff/tools/object_pcd_denoising.py -c $config_cars_reduced -w $weights_cars_reduced -n car_gen_reduced_1_train -t recreate -cls vehicle.car -s train
echo "Command completed"

echo "Running recreation for cars reduced dataset using condition from val"
python lidiff/tools/object_pcd_denoising.py -c $config_cars_reduced -w $weights_cars_reduced -n car_gen_reduced_1_val -t recreate -cls vehicle.car -s val
echo "Command completed"

config_cars_reduced="$config_dir/config_car_gen_reduced_points.yaml"
weights_cars_reduced="$weights_dir/car_gen_point_limited_1_epoch=99.ckpt"

echo "Running recreation for cars subsampled points dataset using condition from train"
python lidiff/tools/object_pcd_denoising.py -c $config_cars_reduced -w $weights_cars_reduced -n car_gen_subsampled_1 -t recreate -cls vehicle.car -s train
echo "Command completed"

echo "Running recreation for cars subsampled points dataset using condition from val"
python lidiff/tools/object_pcd_denoising.py -c $config_cars_reduced -w $weights_cars_reduced -n car_gen_subsampled_1 -t recreate -cls vehicle.car -s val
echo "Command completed"