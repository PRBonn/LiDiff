import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from lidiff.datasets.dataloader.NuscenesObjects import NuscenesObjectsSet
from diffusers import DPMSolverMultistepScheduler
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME
import lidiff.datasets.datasets_objects as datasets_objects
import lidiff.models.models_objects_full_diff as models_objects
import open3d as o3d
import os
from tqdm import tqdm

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def p_sample_loop(model: models_objects.DiffusionPoints, x_t, x_cond, x_uncond, batch_indices):
    model.scheduler_to_cuda()

    for t in tqdm(range(len(model.dpm_scheduler.timesteps))):
        random_ints = torch.ones(x_cond.shape[0]).cuda().long() * model.dpm_scheduler.timesteps[t].cuda()
        t = random_ints[batch_indices] 

        with torch.no_grad():
            noise_t = model.classfree_forward(x_t, x_cond, x_uncond, t).squeeze(1)
            torch.cuda.empty_cache()
        input_noise = x_t.F

        x_t = model.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
        
        x_t = model.points_to_tensor(x_t, batch_indices)

        torch.cuda.empty_cache()

    return x_t.F

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
def main(config, weights):
    set_deterministic()

    cfg = yaml.safe_load(open(config))

    #Load data and model
    model = models_objects.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
    print(model.hparams)

    collate = datasets_objects.NuscenesObjectCollator(coordinate_type=cfg['data']['coordinates'])

    data_set = NuscenesObjectsSet(
            data_dir=cfg['data']['data_dir'], 
            split='train', 
            points_per_object=cfg['data']['points_per_object']
        )
    loader = DataLoader(data_set, batch_size=cfg['train']['batch_size'], shuffle=True,
                        num_workers=cfg['train']['num_workers'], collate_fn=collate)
    

    model.cuda()
    model.eval()
    for batch_index, batch in tqdm(enumerate(loader)):
        model.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=model.t_steps,
                beta_start=model.hparams['diff']['beta_start'],
                beta_end=model.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        model.dpm_scheduler.set_timesteps(model.s_steps)
        model.scheduler_to_cuda()

        x_object = batch['pcd_object']

        x_center = batch['center']
        x_size = batch['size']
        x_orientation = batch['orientation']

        batch_indices = batch['batch_indices'].to(model.device)

        new_cyl = x_center.clone()
        new_cyl[:,0] = new_cyl[:,0] * -1
        x_cond = torch.cat((torch.hstack((new_cyl[:,0][:, None], x_orientation)), torch.hstack((new_cyl[:,1:], x_size))),-1).to(model.device)
        x_uncond = torch.zeros_like(x_cond).to(model.device)
        x_t = torch.randn(x_object.shape, device=model.device)
        x_t = model.points_to_tensor(x_t, batch_indices)
        print("Generating flipped phi angle")
        x_gen_flipped_phi = p_sample_loop(model, x_t, x_cond, x_uncond, batch_indices)
        
        new_yaw = x_orientation.clone()
        new_yaw *= -1
        x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], new_yaw)), torch.hstack((x_center[:,1:], x_size))),-1).to(model.device)
        x_uncond = torch.zeros_like(x_cond).to(model.device)
        x_t = torch.randn(x_object.shape, device=model.device)
        x_t = model.points_to_tensor(x_t, batch_indices)
        print("Generating flipped yaw angle")
        x_gen_flipped_yaw = p_sample_loop(model, x_t, x_cond, x_uncond, batch_indices)

        print("Saving point clouds")
        curr_index = 0
        for pcd_index in range(batch['num_points'].shape[0]):
            max_index = int(curr_index + batch['num_points'][pcd_index].item())
            cond = x_cond[pcd_index]
            generated_flipped_phi = x_gen_flipped_phi[curr_index:max_index]
            generated_flipped_yaw = x_gen_flipped_yaw[curr_index:max_index]
            
            np.savetxt(f'/home/ekirby/scania/ekirby/datasets/augmented_cars_from_nuscenes/train/car_{batch_index}_{pcd_index}_phi_flipped', generated_flipped_phi.cpu().detach().numpy())
            np.savetxt(f'/home/ekirby/scania/ekirby/datasets/augmented_cars_from_nuscenes/train/car_{batch_index}_{pcd_index}_yaw_flipped', generated_flipped_yaw.cpu().detach().numpy())
            np.savetxt(f'/home/ekirby/scania/ekirby/datasets/augmented_cars_from_nuscenes/train/car_{batch_index}_{pcd_index}_condition', cond.cpu().detach().numpy())
            
if __name__ == "__main__":
    main()