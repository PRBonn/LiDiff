import torch
import torch.nn.functional
import torch.utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
from lidiff.datasets.dataloader.ShapeNetObjects import ShapeNetObjectsSet
from lidiff.utils import data_map
import numpy as np

warnings.filterwarnings('ignore')

__all__ = ['ShapeNetObjectsDataModule']

class ShapeNetObjectsDataModule(LightningDataModule):
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
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='train',
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'], shuffle=True,
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def val_dataloader(self, pre_training=True):
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val'
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                            num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

    def test_dataloader(self):
        collate = ShapeNetObjectCollator()

        data_set = ShapeNetObjectsSet(
                data_dir=self.cfg['data']['data_dir'], 
                split='val',
            )
        loader = DataLoader(data_set, batch_size=self.cfg['train']['batch_size'],
                             num_workers=self.cfg['train']['num_workers'], collate_fn=collate)
        return loader

class ShapeNetObjectCollator:
    def __init__(self):
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
        
        center = torch.from_numpy(np.stack(batch[1])).float()
        orientation = torch.from_numpy(np.stack(batch[3])).float()

        class_mapping = torch.tensor([data_map.class_mapping[class_name] for class_name in batch[6]]).reshape(-1, 1)
        class_mapping = torch.nn.functional.one_hot(class_mapping, num_classes=3)

        return {'pcd_object': pcd_object, 
            'center': center,
            'size': torch.stack(batch[2]).float(),
            'orientation': orientation,
            'batch_indices': batch_indices,
            'num_points': num_points_tensor,
            'class': class_mapping
        }

dataloaders = {
    'shapenet': ShapeNetObjectsDataModule,
}

