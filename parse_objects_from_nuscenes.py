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


def parse_objects_from_nuscenes(points_threshold, object_name, object_tag):
    # Path to the dataset
    dataroot = '/datasets_local/nuscenes'

    # Initialize the nuScenes class
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    # Directory to save extracted data
    output_dir = f'/home/ekirby/scania/ekirby/datasets/{object_name}_from_nuscenes'
    os.makedirs(output_dir, exist_ok=True)

    train_split = set(splits.train)

    object_lidar_data = {'train':[], 'val':[]}
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
            if object_tag in object.name:
                annotation = nusc.get('sample_annotation', object.token)
                num_lidar_points = annotation['num_lidar_pts']
                if num_lidar_points < points_threshold:
                    continue
                object_info = {
                    'instance_token': object.token,
                    'sample_token': sample_token,
                    'scene_token': scene_token,
                    'sample_data_lidar_token': sample_data_lidar_token,
                    'lidar_data_filepath': lidar_filepath,
                    'class': object_tag,
                    'center': object.center.tolist(),
                    'size': object.wlh.tolist(),
                    'rotation_real': object.orientation.real.tolist(),
                    'rotation_imaginary': object.orientation.imaginary.tolist(),
                }
                object_lidar_data[split].append(object_info)


    print(f"After parsing, {len(object_lidar_data['train'])} objects in train, {len(object_lidar_data['val'])} in val")
    with open(f'{output_dir}/{object_name}_from_nuscenes_train_val.json', 'w') as fp:
        json.dump(object_lidar_data, fp)

if __name__ == '__main__':
    points_threshold = 50
    object_name = 'motorcycles'
    object_tag = 'vehicle.motorcycle'
    parse_objects_from_nuscenes(points_threshold, object_name, object_tag)