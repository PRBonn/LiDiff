import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import collate_reconstruction_scans
from natsort import natsorted
import os
import numpy as np
import yaml
import time

import warnings

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class ReconstructionKittiSet(Dataset):
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, downsampling, condition_point_ratio, dataset_norm=False, std_axis_norm=False, n_data=None):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range

        self.split = split
        self.seqs = seqs

        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

        if n_data == None:
            self.nr_data = len(self.points_datapath)
        else:
            self.nr_data = n_data

        self.downsampling = downsampling
        self.condition_point_ratio = condition_point_ratio

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def datapath_list(self):
        self.points_datapath = []

        for seq in self.seqs:
            point_seq_path = os.path.join(self.data_dir, 'dataset', 'sequences', seq)
            point_seq_bin = natsorted(os.listdir(os.path.join(point_seq_path, 'velodyne')))
 
            for file_num in range(0, len(point_seq_bin)):
                self.points_datapath.append(os.path.join(point_seq_path, 'velodyne', point_seq_bin[file_num]))

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def __getitem__(self, index):
        p_scan = np.fromfile(self.points_datapath[index], dtype=np.float32)
        p_scan = p_scan.reshape((-1,4))[:,:3]

        if self.split != 'test':
            label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF # Get the label from the least significant 16 bits

        dist_part = np.sum(p_scan**2, -1)**.5 # Get total magnitude of distance
        p_scan = p_scan[(dist_part < self.max_range) & (dist_part > 3.5)]
        p_scan = p_scan[p_scan[:,2] > -4.] # Filter out noisy points

        p_part = p_scan

        if self.split == 'train':
            p_concat = np.concatenate((p_scan, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_scan = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]

        n_part = int(self.num_points * self.condition_point_ratio)

        return collate_reconstruction_scans(
            p_scan,
            p_part,
            self.num_points,
            n_part,
            self.resolution,
            self.points_datapath[index],
            p_mean=self.data_stats['mean'],
            p_std=self.data_stats['std'],
            downsampling=self.downsampling
        )

    def __len__(self):
        return self.nr_data

##################################################################################################
