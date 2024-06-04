import os
import json
from cv2 import threshold
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix, points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits
from tqdm import tqdm


def parse_cars_from_nuscenes(points_threshold):
    # Path to the dataset
    dataroot = '/datasets_local/nuscenes'

    # Initialize the nuScenes class
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    # Directory to save extracted data
    output_dir = '/home/ekirby/scania/ekirby/datasets/cars_from_nuscenes'
    os.makedirs(output_dir, exist_ok=True)

    train_split = set(splits.train)

    # Loop through scenes and extract car annotations and LiDAR data
    car_lidar_data = {'train':[], 'val':[]}
    for sample in tqdm(nusc.sample):
        scene_token = sample['scene_token']
        sample_token = sample['token']
        sample_data_lidar_token = sample['data']['LIDAR_TOP']
        scene_name = nusc.get('scene', scene_token)['name']
        split = 'train' if scene_name in train_split else 'val'

        objects = nusc.get_sample_data(sample_data_lidar_token)[1]
        lidar_data = nusc.get('sample_data', sample_data_lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])

        for object in objects:
            if 'vehicle.car' in object.name:
                annotation = nusc.get('sample_annotation', object.token)
                num_lidar_points = annotation['num_lidar_pts']
                if num_lidar_points < points_threshold:
                    continue
                car_info = {
                    'instance_token': object.token,
                    'sample_token': sample_token,
                    'scene_token': scene_token,
                    'sample_data_lidar_token': sample_data_lidar_token,
                    'lidar_data_filepath': lidar_filepath,
                    'class': 'car',
                    'center': object.center.tolist(),
                    'size': object.wlh.tolist(),
                    'rotation_real': object.orientation.real.tolist(),
                    'rotation_imaginary': object.orientation.imaginary.tolist(),
                }
                car_lidar_data[split].append(car_info)


    with open(f'{output_dir}/cars_from_nuscenes_train_val.json', 'w') as fp:
        json.dump(car_lidar_data, fp)

if __name__ == '__main__':
    points_threshold = 10
    parse_cars_from_nuscenes(points_threshold)