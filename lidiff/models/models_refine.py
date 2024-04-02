import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs
from pytorch3d.loss import chamfer_distance

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall

class RefineDiffusion(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # learn N offsets per point: out_channel is 3 * N
        self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3*self.hparams['train']['up_factor'])

        n_part = int(self.hparams['data']['num_points'] / self.hparams['data']['scan_window'])
        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(0.001,0.01,100)
    
    def points_to_tensor(self, x_feats, mean, std):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def forward_refine(self, x):
        return self.model_refine(x)

    def training_step(self, batch, batch_idx):
        x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
        x_coord = x_feats.clone()
        x_coord = torch.round(x_feats / self.hparams['data']['resolution'])

        x_feats = x_feats[:,1:]

        x_t = ME.TensorField(
            features=x_feats,
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
        refine_upsample_pcd = x_feats[:,None,:] + offset
        refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

        loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
        self.log('train/cd_loss', loss)
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.hparams['data']['resolution'])
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
            refine_upsample_pcd = x_feats[:,None,:] + offset
            refine_upsample_pcd = refine_upsample_pcd.reshape(batch['pcd_full'].shape[0],-1,3)
    
            loss, _ = chamfer_distance(refine_upsample_pcd, torch.tensor(batch['pcd_full']))
            self.log('val/cd_loss', loss)
            torch.cuda.empty_cache()
    
            return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x_feats = ME.utils.batched_coordinates(list(batch['pcd_noise']), dtype=torch.float32, device=self.device)
            x_coord = x_feats.clone()
            x_coord = torch.round(x_feats / self.hparams['data']['resolution'])
    
            x_feats = x_feats[:,1:]
    
            x_t = ME.TensorField(
                features=x_feats,
                coordinates=x_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=self.device,
            )
    
            offset = self.forward_refine(x_t).reshape(-1,self.hparams['train']['up_factor'],3)
            refine_pcd = x_feats[:,None,:] + offset
            refine_pcd = refine_pcd.reshape(batch['pcd_full'].shape[0],-1,3)

            pcd_refine = o3d.geometry.PointCloud()
            pcd_refine.points = o3d.utility.Vector3dVector(refine_pcd[0].cpu().numpy())
            pcd_refine.paint_uniform_color([1.,.2,.2])
            pcd_refine.estimate_normals()
            o3d.visualization.draw_geometries([pcd_refine])
    
            loss, _ = chamfer_distance(refine_pcd, torch.tensor(batch['pcd_full']))
            self.log('test/cd_loss', loss)
            torch.cuda.empty_cache()

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))

        return optimizer

#######################################
# Modules
#######################################
