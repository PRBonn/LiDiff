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
import lidiff.utils.data_map as data_map
import sys


def parse_objects_from_nuscenes(points_threshold, object_name, object_tag, range_to_use):
    # Path to the dataset
    dataroot = '/datasets_local/nuscenes'

    # Initialize the nuScenes class
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    # Directory to save extracted data
    output_dir = f'/home/ekirby/scania/ekirby/datasets/{object_name}_from_nuscenes'
    os.makedirs(output_dir, exist_ok=True)

    train_split = set(splits.train)

    object_lidar_data = {'train':[], 'val':[]}
    for i in tqdm(range_to_use):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']
        sample_token = sample['token']
        sample_data_lidar_token = sample['data']['LIDAR_TOP']
        scene_name = nusc.get('scene', scene_token)['name']
        split = 'train' if scene_name in train_split else 'val'

        objects = nusc.get_sample_data(sample_data_lidar_token)[1]
        lidar_data = nusc.get('sample_data', sample_data_lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])
        lidarseg_label_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', sample_data_lidar_token)['filename'])
        points = np.fromfile(lidar_filepath, dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        labels = np.fromfile(lidarseg_label_filename, dtype=np.uint8).reshape((-1, 1))
        for object in objects:
            if object_tag in object.name:
                points_from_object = points_in_box(object, points=points[:,:3].T)
                object_points = points[points_from_object][:,:3]
                object_points = object_points[(labels[points_from_object] == data_map.class_mapping[object.name]).flatten()]
                
                num_lidar_points = len(object_points)
                if num_lidar_points < points_threshold:
                    continue

                object_info = {
                    'instance_token': object.token,
                    'sample_token': sample_token,
                    'scene_token': scene_token,
                    'sample_data_lidar_token': sample_data_lidar_token,
                    'lidar_data_filepath': lidar_filepath,
                    'lidarseg_label_filepath': lidarseg_label_filename,
                    'class': object.name,
                    'center': object.center.tolist(),
                    'size': object.wlh.tolist(),
                    'rotation_real': object.orientation.real.tolist(),
                    'rotation_imaginary': object.orientation.imaginary.tolist(),
                    'num_lidar_points': num_lidar_points,
                }
                object_lidar_data[split].append(object_info)

    print(f"After parsing, {len(object_lidar_data['train'])} objects in train, {len(object_lidar_data['val'])} in val")
    with open(f'{output_dir}/{object_name}_from_nuscenes_train_val.json', 'w') as fp:
        json.dump(object_lidar_data, fp)

    return object_lidar_data, output_dir

def parse_largest_x_from_dataset(output_dir, object_name, object_lidar_data, top_x):
    reduced_train_val_objects = {'train':[], 'val':[]}
    train_objects = object_lidar_data['train']
    val_objects = object_lidar_data['val']
    print("Sorting object lidar data")
    train_sorted_by_num_points = sorted(train_objects, key=lambda x: x['num_lidar_points'], reverse=True)
    val_sorted_by_num_points = sorted(val_objects, key=lambda x: x['num_lidar_points'], reverse=True)
    print("Taking top x from object lidar data")
    reduced_train_val_objects['train'] = train_sorted_by_num_points[:top_x]
    reduced_train_val_objects['val'] = val_sorted_by_num_points[:top_x]

    with open(f'{output_dir}/{object_name}_from_nuscenes_train_val_reduced.json', 'w') as fp:
        json.dump(reduced_train_val_objects, fp)

if __name__ == '__main__':
    range_index = int(sys.argv[1])
    points_threshold = 20
    object_name = f'all_objects_filtered_range_{range_index}'
    object_tag = ''
    ranges = [range(0,8537), range(8537,17074), range(17074,25612), range(25612,34149)]
    object_lidar_data, output_dir = parse_objects_from_nuscenes(points_threshold, object_name, object_tag, ranges[range_index])