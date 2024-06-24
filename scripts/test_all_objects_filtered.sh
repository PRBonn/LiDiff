#!/bin/bash

cd /home/ekirby/workspace/LiDiff

# Directory containing the config files
config_dir="lidiff/config/object_generation"
weights_dir="lidiff/checkpoints"

config="$config_dir/config_filtered_gen_test.yaml"
weights="$weights_dir/all_objects_gen_filtered_1_epoch=99.ckpt"

echo "Recreation cars"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_filtered_cars_train_1 -t recreate -cls vehicle.car -s train -e 1
echo "Command completed"

echo "Recreation bikes"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_filtered_bikes_train_1 -t recreate -cls vehicle.bicycle -s train -e 1
echo "Command completed"

echo "Recreation motos"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_filtered_motos_train_1 -t recreate -cls vehicle.motorcycle -s train -e 1
echo "Command completed"

echo "Interpolation cars"
python lidiff/tools/object_pcd_denoising.py -c $config -w $weights -n all_objects_filtered_cars_train_2 -t interpolate -cls vehicle.car -s train -e 2
echo "Command completed"