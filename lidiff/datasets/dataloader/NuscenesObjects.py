import os
import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset
import json
import numpy as np
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from lidiff.utils.three_d_helpers import extract_yaw_angle, cartesian_to_spherical

class NuscenesObjectsSet(Dataset):
    def __init__(self, data_dir, split, points_per_object=None, volume_expansion=1., recenter=True, ):
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
        rings = torch.from_numpy(points[points_from_object])[:,4].long()
        ring_indexes = torch.nn.functional.one_hot(rings, num_classes=32)
        num_points = object_points.shape[0]

        if self.do_recenter:
            object_points -= center

        return [object_points, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name]