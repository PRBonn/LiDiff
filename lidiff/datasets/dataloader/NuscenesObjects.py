import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import json
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from lidiff.utils import data_map
from lidiff.utils.three_d_helpers import extract_yaw_angle, cartesian_to_spherical
import open3d as o3d
from nuscenes.utils.data_io import load_bin_file
from lidiff.utils.three_d_helpers import cartesian_to_cylindrical

class NuscenesObjectsSet(Dataset):
    def __init__(
            self, 
            data_dir, 
            split, 
            points_per_object=None, 
            volume_expansion=1., 
            recenter=True, 
            align_objects=False, 
            relative_angles=False,
            excluded_tokens=None,
            permutation=[],
        ):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]
        if isinstance(self.data_index, dict):
            if excluded_tokens != None:
                print(f'Before existing object filtering: {len(self.data_index)} objects')
                self.data_index = [value for key, value in self.data_index.items() if key not in excluded_tokens]
                print(f'After existing object filtering: {len(self.data_index)} objects')
            else:
                self.data_index = list(self.data_index.values())
    
        if len(permutation) > 0:
            print(f'Limiting dataset to {len(permutation)} samples')
            self.data_index = [self.data_index[i] for i in permutation]
            print(f'After limiting, length of dataset is {len(self.data_index)}')
            
        self.nr_data = len(self.data_index)
        self.points_per_object = points_per_object
        self.volume_expansion = volume_expansion
        self.do_recenter = recenter
        self.align_objects = align_objects
        self.relative_angles = relative_angles

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = object_json['class']
        points = np.fromfile(object_json['lidar_data_filepath'], dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        mask = np.load(object_json['object_sample_index'])
        center = np.array(object_json['center'])
        size = np.array(object_json['size'])
        rotation_real = np.array(object_json['rotation_real'])
        rotation_imaginary = np.array(object_json['rotation_imaginary'])
        orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)

        object_points = points[mask][:,:3]

        if self.points_per_object > 0:
            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(object_points)

            if object_points.shape[0] > self.points_per_object:
                pcd_object = pcd_object.farthest_point_down_sample(self.points_per_object)
            
            object_points = torch.tensor(np.array(pcd_object.points))
            concat_part = int(np.ceil(self.points_per_object / object_points.shape[0]) )
            object_points = object_points.repeat(concat_part, 1)
            object_points = object_points[torch.randperm(object_points.shape[0])][:self.points_per_object]
        
        num_points = object_points.shape[0]
        ring_indexes = torch.zeros(object_points.shape[0])
        if self.do_recenter:
            object_points -= center
        
        center = cartesian_to_cylindrical(center[None,:])[0]
        yaw = orientation.yaw_pitch_roll[0]
        
        if self.align_objects:
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            object_points = np.dot(object_points, rotation_matrix.T)

        if self.relative_angles:
            center[0] -= yaw
        
        return [object_points, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name, object_json['instance_token']]