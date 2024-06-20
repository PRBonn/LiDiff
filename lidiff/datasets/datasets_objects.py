import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from lidiff.datasets.dataloader.NuscenesObjects import NuscenesObjectsSet
import warnings
from lidiff.utils import data_map
from lidiff.utils.three_d_helpers import cartesian_to_cylindrical
import numpy as np

warnings.filterwarnings('ignore')

__all__ = ['NuscenesObjectsDataModule']

class NuscenesObjectsDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):
        collate = NuscenesObjectCollator(coordinate_type=self.cfg['data']['coordinates'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train', 
                points_per_object=self.cfg['data']['points_per_object']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = NuscenesObjectCollator(coordinate_type=self.cfg['data']['coordinates'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
                points_per_object=self.cfg['data']['points_per_object']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = NuscenesObjectCollator(coordinate_type=self.cfg['data']['coordinates'])

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val', 
                points_per_object=self.cfg['data']['points_per_object']
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class NuscenesObjectCollator:
    def __init__(self, mode='diffusion', coordinate_type='standard'):
        self.mode = mode
        self.coordinate_type = coordinate_type
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))
        pcd_object = torch.from_numpy(np.vstack(batch[0]))
        batch_indices = torch.zeros(pcd_object.shape[0])

        num_points_tensor = torch.Tensor(batch[4])
        cumulative_indices = torch.cumsum(num_points_tensor, dim=0).long()
        batch_indices[cumulative_indices-1] = 1
        batch_indices = batch_indices.cumsum(0).long()
        batch_indices[-1] = batch_indices[-2]

        # center = torch.from_numpy(np.stack(batch[1])).float()
        
        # if self.coordinate_type == 'cartesian':
        #     center = torch.from_numpy(np.stack(batch[1])).float()
        #     orientation = torch.Tensor([[quaternion.angle] for quaternion in batch[3]]).float()
        # elif self.coordinate_type == 'cylindrical':
        #     center = cartesian_to_cylindrical(np.stack(batch[1]))
        #     orientation = np.array([[quaternion.yaw_pitch_roll[0]] for quaternion in batch[3]])
        #     phi, orientation = center[:, 0].flatten(), orientation.flatten()
        #     center[:,0] = phi
        #     center = torch.from_numpy(center).float()
        #     orientation = torch.from_numpy(orientation).float()[:, None]
            
        class_mapping = torch.tensor([data_map.class_mapping[class_name] for class_name in batch[6]]).reshape(-1, 1)
        num_classes = max(data_map.class_mapping.values()) + 1
        class_mapping = torch.nn.functional.one_hot(class_mapping, num_classes=num_classes)

        return {'pcd_object': pcd_object, 
            'center':  torch.from_numpy(np.vstack(batch[1])).float(),
            'size':  torch.from_numpy(np.vstack(batch[2])).float(),
            'orientation': torch.zeros((num_points_tensor.shape[0], 1)),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'ring_indexes': torch.vstack(batch[5]),
            'class': class_mapping
        }

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
}

