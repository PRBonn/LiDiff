#!/bin/bash

cd /home/ekirby/workspace/LiDiff

# Directory containing the config files
config_dir="lidiff/config/object_generation"
weights_dir="lidiff/checkpoints"

config_shapenet_motos="$config_dir/config_shapenet_motos.yaml"
weights_shapenet_motos="$weights_dir/shapnet_moto_gen_1_epoch=99.ckpt"

echo "Running recreation for motorcycle from shapenet condition from train"
python lidiff/tools/object_pcd_denoising.py -c $config_shapenet_motos -w $weights_shapenet_motos -n shapenet_motos_1_train -t recreate -cls vehicle.motorcycle -s train
echo "Command completed"