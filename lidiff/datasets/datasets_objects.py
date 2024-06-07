import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from lidiff.datasets.dataloader.NuscenesObjects import NuscenesObjectsSet
import warnings
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
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train', 
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val', 
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = NuscenesObjectCollator()

        data_set = NuscenesObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val', 
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class NuscenesObjectCollator:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))
        pcd_object = torch.vstack(batch[0])
        batch_indices = torch.zeros(pcd_object.shape[0])

        num_points_tensor = torch.Tensor(batch[4])
        cumulative_indices = torch.cumsum(num_points_tensor, dim=0).long()
        batch_indices[cumulative_indices-1] = 1
        batch_indices = batch_indices.cumsum(0).long()
        batch_indices[-1] = batch_indices[-2]
        spherical_coordinates = torch.from_numpy(cartesian_to_cylindrical(np.stack(batch[1]))).float()
        return {'pcd_object': pcd_object, 
            'center': spherical_coordinates,
            'size': torch.stack(batch[2]).float(),
            'orientation': torch.stack(batch[3]).float(),
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'ring_indexes': torch.vstack(batch[5]),
        }

dataloaders = {
    'nuscenes': NuscenesObjectsDataModule,
}

