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
import open3d as o3d
import glob

class ShapeNetObjectsSet(Dataset):
    def __init__(self, data_dir, split):
        super().__init__()
        self.data_dir = f'{data_dir}/{split}'
        self.files = glob.glob(f'{self.data_dir}/**.npy')
        self.nr_data = len(self.files)

    def __len__(self):
        return self.nr_data
    
    def __getitem__(self, index):
        object_points = np.load(self.files[index])

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
        rotation_matrix_pitch = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        rotated_point_cloud = object_points.dot(rotation_matrix_pitch.T)
        size = np.zeros(3)
        for i in range(3):
            size[i] = np.max(object_points[i]) - np.min(object_points[i])
        center = np.zeros(3)
        orientation = np.zeros(1)
        ring_indexes = np.zeros_like(object_points)
        class_name = 'vehicle.motorcycle'


        return [object_points, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name]