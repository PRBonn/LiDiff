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

class NuscenesObjectsSet(Dataset):
    def __init__(self, data_dir, split, points_per_object=None, volume_expansion=1., recenter=True):
        super().__init__()
        with open(data_dir, 'r') as f:
            self.data_index = json.load(f)[split]

        self.nr_data = len(self.data_index)
        self.points_per_object = points_per_object
        self.volume_expansion = volume_expansion
        self.do_recenter = recenter

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_json = self.data_index[index]
        
        class_name = object_json['class']
        points = np.fromfile(object_json['lidar_data_filepath'], dtype=np.float32).reshape((-1, 5)) #(x, y, z, intensity, ring index)
        center = np.array(object_json['center'])
        size = np.array(object_json['size'])
        rotation_real = np.array(object_json['rotation_real'])
        rotation_imaginary = np.array(object_json['rotation_imaginary'])

        orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
        box = Box(center=center, size=size, orientation=orientation)
        
        points_from_object = points_in_box(box, points=points[:,:3].T, wlh_factor=self.volume_expansion)
        object_points = torch.from_numpy(points[points_from_object])[:,:3]

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
        ring_indexes = torch.zeros_like(object_points)
        if self.do_recenter:
            object_points -= center

        return [object_points, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name]