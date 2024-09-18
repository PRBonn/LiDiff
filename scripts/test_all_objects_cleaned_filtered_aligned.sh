#!/bin/bash

cd /home/ekirby/workspace/LiDiff

# Directory containing the config files
config_dir="lidiff/config/object_generation"
weights_dir="lidiff/checkpoints"

config="$config_dir/config_all_objects_cleaned_aligned_eval.yaml"
weights="$weights_dir/all_objects_cleaned_aligned_1_epoch=499.ckpt"

# echo "Recreation cars"
# python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_cleaned_aligned_cars_train_1 -t recreate -cls vehicle.car -s train -e 2
# echo "Command completed"

echo "Interpolqtion bikes"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_cleaned_aligned_bikes_train_2 -t interpolate -cls vehicle.bicycle -s train -e 2
echo "Command completed"

echo "Interpolqtion motos"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_cleaned_aligned_motos_train_2 -t interpolate -cls vehicle.motorcycle -s train -e 2
echo "Command completed"

# echo "Interpolation cars"
# python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_cleaned_aligned_cars_train_2 -t interpolate -cls vehicle.car -s train -e 2
# echo "Command completed"