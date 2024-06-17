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
        size = np.zeros(3)
        for i in range(3):
            size[i] = np.max(object_points[i]) - np.min(object_points[i])
        center = np.zeros(3)
        orientation = np.zeros(1)
        num_points = object_points.shape[0]
        ring_indexes = np.zeros_like(object_points)
        class_name = 'vehicle.motorcycle'

        return [object_points, center, torch.from_numpy(size), orientation, num_points, ring_indexes, class_name]