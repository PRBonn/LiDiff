import click
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
from lidiff.datasets.datasets_objects import NuscenesObjectsDataModule
from lidiff.models.models_objects_full_diff import DiffusionPoints
from lidiff.utils.three_d_helpers import cylindrical_to_cartesian, angle_add
from diffusers import DPMSolverMultistepScheduler
from torch.multiprocessing import spawn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from nuscenes.utils.data_classes import Quaternion
import json
import glob

def inverse_scale_intensity(scaled_data):
    max_intensity = np.log1p(255.0)
    data_log_transformed = scaled_data * max_intensity
    original_data = np.round(np.clip(np.expm1(data_log_transformed), a_max=255.0, a_min=0.0))
    return original_data

def realign_pointclouds_to_scan(x_gen, orientation, center, aligned_angle):
    cos_yaw = np.cos(orientation)
    sin_yaw = np.sin(orientation)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    x_gen[:, :3] = np.dot(x_gen[:, :3], rotation_matrix.T)
    new_center = center.copy()
    new_center[0] = angle_add(aligned_angle, orientation)
    new_center = cylindrical_to_cartesian(new_center[None, :]).squeeze(0)
    x_gen[:, :3] += new_center
    x_gen[:, 3] = inverse_scale_intensity(x_gen[:, 3])  # rescale intensity to 0-30
    return x_gen

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default='config/ablation_7/xs_1a_logen_bikes_cleaned_gen.yaml')
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default='checkpoints/xs_1a_logen_bikes_cleaned/last.ckpt')
@click.option('--split',
              '-s',
              type=str,
              help='train or val split',
              default='train')
@click.option('--rootdir',
              '-r',
              type=str,
               default='/home/ekirby/scania/ekirby/datasets/replaced_nuscenes_datasets_TEST')
@click.option('--token_to_data', type=str, default='/home/ekirby/scania/ekirby/datasets/all_objects_nuscenes_cleaned/all_objects_cleaned_token_to_data.json')
@click.option('--consistent_seed', type=bool, default=True)
@click.option('--class_name', '-cls', type=str)
@click.option('--permutation_file', type=str, default=None)
@click.option('--limit_samples_count', type=int, default=-1)
def main(config, weights, split, rootdir, token_to_data, consistent_seed, class_name, permutation_file, limit_samples_count):
    cfg = yaml.safe_load(open(config))
    batch_size = cfg['train']['batch_size']
    world_size = cfg['train']['n_gpus']
    experiment_dir = cfg['experiment']['id']
    cfg['data']['gen_class_name'] = class_name
    existing_paths = glob.glob(f'{rootdir}/{experiment_dir}/*/{cfg["data"]["gen_class_name"]}/**')
    existing_tokens = set()
    for path in existing_paths:
        existing_tokens.add(path.split('/')[-1])

    # Spawn a process for each GPU
    spawn(train, args=(world_size, cfg, weights, split, rootdir, batch_size, token_to_data, consistent_seed, existing_tokens, permutation_file, limit_samples_count), nprocs=world_size, join=True)

def train(rank, world_size, cfg, weights, split, rootdir, batch_size, token_to_data, consistent_seed, existing_tokens, permutation_file, limit_samples_count):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    with open(token_to_data, 'r') as f:
        token_to_data_in_split = json.load(f)[split]

    experiment_dir = cfg['experiment']['id']

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    permutation = []
    if permutation_file != None and limit_samples_count > 0:
        with open(permutation_file, 'r') as f:
            permutation = json.load(f)[split]
        total_samples = min(limit_samples_count, len(permutation))
        permutation = np.array(permutation)[:total_samples]
    dataset, collate = NuscenesObjectsDataModule(cfg).build_dataset(split, existing_tokens, permutation)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=cfg['train']['num_workers'], collate_fn=collate)

    model.eval()

    for _, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            x_object = batch['pcd_object']
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            batch_indices = batch['batch_indices'].cuda()
            num_points = batch['num_points']
            annotation_tokens = batch['tokens']
            x_cond = torch.cat((x_center, x_size),-1)

            model.module.dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=model.module.t_steps,
                    beta_start=model.module.hparams['diff']['beta_start'],
                    beta_end=model.module.hparams['diff']['beta_end'],
                    beta_schedule='linear',
                    algorithm_type='sde-dpmsolver++',
                    solver_order=2,
                )
            model.module.dpm_scheduler.set_timesteps(model.module.s_steps)
            model.module.scheduler_to_cuda()

            x_t = torch.randn(x_object.shape, device=model.device)

            x_cond = x_cond.cuda()
            x_uncond = torch.zeros_like(x_cond, device=x_cond.device)
            x_t = model.module.points_to_tensor(x_t, batch_indices)
            x_generated = model.module.p_sample_loop(x_t, x_cond, x_uncond, batch_indices, num_points, batch['class'].cuda()).F.cpu().detach().numpy()
            x_object_points = x_object.cpu().detach().numpy()

            curr_index = 0
            for pcd_index in range(batch['num_points'].shape[0]):
                max_index = int(curr_index + batch['num_points'][pcd_index].item())

                x_gen = x_generated[curr_index:max_index]
                x_org = x_object_points[curr_index:max_index]
                
                curr_index = max_index
                
                center = x_center[pcd_index].cpu().detach().numpy()
                orientation = x_orientation[pcd_index].cpu().detach().numpy()

                token = annotation_tokens[pcd_index]
                sample_data = token_to_data_in_split[token]
                
                size = np.array(sample_data['size'])
                rotation_real = np.array([sample_data['rotation_real']])
                rotation_imaginary = np.array(sample_data['rotation_imaginary'])
                orientation = np.array([Quaternion(real=rotation_real, imaginary=rotation_imaginary).yaw_pitch_roll[0]])
                conditions = np.concatenate((center, size, orientation, rotation_real, rotation_imaginary))

                sample_token = sample_data['sample_token']
                sample_dir = f'{rootdir}/{experiment_dir}/{sample_token}/{cfg["data"]["gen_class_name"]}/{token}'
                
                x_gen = realign_pointclouds_to_scan(x_gen, orientation.item(), center, center[0])
                x_org = realign_pointclouds_to_scan(x_org, orientation.item(), center, center[0])
                
                instance_index = 0
                os.makedirs(sample_dir, exist_ok=True)
                np.savetxt(f'{sample_dir}/generated_{instance_index}.txt', x_gen)
                np.savetxt(f'{sample_dir}/original_{instance_index}.txt', x_org)
                np.savetxt(f'{sample_dir}/conditions_{instance_index}.txt', conditions)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
