#!/bin/bash

cd /home/ekirby/workspace/LiDiff

pip install -e .

cd lidiff

replaced_dir=/home/ekirby/scania/ekirby/datasets/replaced_nuscenes_datasets

# Generate 1000 BARRIERS, train and val
config=config/sparse_cnn_ablation/scnn_barriers_gen.yaml
weights=checkpoints/scnn_barriers/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/barriers_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 BIKES, train and val
config=config/sparse_cnn_ablation/scnn_bikes_cleaned_gen.yaml
weights=checkpoints/scnn_bikes_cleaned/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/bikes_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 BUSES, train and val
config=config/sparse_cnn_ablation/scnn_bus_gen.yaml
weights=checkpoints/scnn_bus/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/buses_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 CARS, train and val
config=config/sparse_cnn_ablation/scnn_cars_gen.yaml
weights=checkpoints/scnn_cars/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/cars_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 CONSTRUCTION VEHICLES, train and val
config=config/sparse_cnn_ablation/scnn_construction_vehicles_gen.yaml
weights=checkpoints/scnn_construction_vehicles/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/construction_vehicles_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 MOTORCYCLES, train and val
config=config/sparse_cnn_ablation/scnn_motorcycles_gen.yaml
weights=checkpoints/scnn_motorcycles/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/motorcycles_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 PEDESTRIANS, train and val
config=config/sparse_cnn_ablation/scnn_pedestrian_gen.yaml
weights=checkpoints/scnn_pedestrian/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/pedestrians_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 TRAFFIC CONES, train and val
config=config/sparse_cnn_ablation/scnn_traffic_cones_gen.yaml
weights=checkpoints/scnn_traffic_cones/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/traffic_cones_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 TRAILERS, train and val
config=config/sparse_cnn_ablation/scnn_trailers_gen.yaml
weights=checkpoints/scnn_trailers/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/trailers_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count

# Generate 1000 TRUCKS, train and val
config=config/sparse_cnn_ablation/scnn_trucks_gen.yaml
weights=checkpoints/scnn_trucks/last.ckpt
permutation_file=/home/ekirby/scania/ekirby/datasets/trucks_from_nuscenes/pre_sampled_token_permutation.json
limit_samples_count=1000
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s train -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count
python tools/object_generation_multi_gpu.py -c $config -w $weights -n 1 -s val   -r $replaced_dir --permutation_file $permutation_file --limit_samples_count $limit_samples_count