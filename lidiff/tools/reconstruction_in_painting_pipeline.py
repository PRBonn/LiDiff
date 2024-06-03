import numpy as np
import MinkowskiEngine as ME
import torch
import lidiff.models.minkunet as minknet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler, RePaintScheduler
from pytorch_lightning.core.module import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
import json
import glob

class InPainter(LightningModule):
    def __init__(self, diff_path, denoising_steps, cond_weight, scheduler_type):
        super().__init__()
        ckpt_diff = torch.load(diff_path)
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
        f"The number of denoising steps cannot be bigger than T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()
        self.load_state_dict(ckpt_diff['state_dict'], strict=False)

        self.partial_enc.eval()
        self.model.eval()
        self.cuda()

        # for fast sampling
        self.hparams['diff']['s_steps'] = denoising_steps
        self.scheduler_type = scheduler_type
        if scheduler_type == 'dpm':
            self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
            )
        elif scheduler_type == 'repaint':
            self.dpm_scheduler = RePaintScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
            )
        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda()

        self.hparams['train']['uncond_w'] = cond_weight
        self.hparams['data']['max_range'] = 50.
        self.w_uncond = self.hparams['train']['uncond_w']
        
        exp_dir = diff_path.split('/')[-1].split('.')[0].replace('=','')  + f'_T{denoising_steps}_s{cond_weight}'
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        if self.scheduler_type == 'dpm':
            self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
            self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
            self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
            self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t                                                                                        

    def reset_partial_pcd(self, x_part, x_uncond):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach())
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(1,-1,3)))

        return x_part, x_uncond

    
    def generate_noise_schedule(self, t_T=250, jump_len=10, jump_n_sample=10):
        jumps = {}
        for j in range(0, t_T - jump_len, jump_len):
            jumps[j] = jump_n_sample - 1
        t = t_T
        ts = []
        while t >= 1:
            ts.append(t_T - t)
            t = t-1
            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_len):
                    t = t + 1
                    ts.append(t_T - t)
        return ts
    
    def preprocess_scan(self, scan, mask):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.hparams['data']['max_range']) & (dist > 3.5)][:,:3]
        mask = mask[(dist < self.hparams['data']['max_range']) & (dist > 3.5)]

        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.voxel_down_sample(10*self.hparams['data']['resolution'])
        cond = torch.tensor(np.array(pcd_scan.points)).to(self.device)
        scan = torch.tensor(scan).to(self.device)

        scan = scan[None,:,:]
        cond = cond[None,:,:]

        return scan, cond, mask

    # We can probably keep this as our scan should have the same shape
    def postprocess_scan(self, completed_scan, input_scan):
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < self.hparams['data']['max_range']]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan
    
    def compute_metrics(self, original_scan, inpainted_scan, mask):
        pcd_pred = o3d.geometry.PointCloud()
        points = inpainted_scan.reshape(-1, 3)[mask.squeeze()==0.]
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.paint_uniform_color([1.0, 0.,0.])

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(original_scan.reshape(-1, 3)[mask.squeeze()==0.])
        pcd_gt.paint_uniform_color([0., 1.,0.])

        self.chamfer_distance.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        return cd_mean, cd_std

    def in_paint_scan(self, scan, mask):
        completed_scans = []
        samples_to_generate = ['normal', 'unconditioned']
        processed_scan, cond, mask = self.preprocess_scan(scan, mask)
        mask = torch.from_numpy(mask).to(self.device)
        x_feats = processed_scan + torch.randn(processed_scan.shape, device=self.device)
        metrics = {}
        for sample_type in samples_to_generate:

            x_full = self.points_to_tensor(x_feats)
            x_uncond = self.points_to_tensor(torch.zeros_like(processed_scan))

            if sample_type == 'normal':
                x_cond_normal = self.points_to_tensor(cond)
                completed_scan = self.in_painting_loop(x_init=processed_scan, x_t=x_full, x_cond=x_cond_normal, x_uncond=x_uncond, mask=mask)
            elif sample_type == 'unconditioned':
                completed_scan = self.in_painting_loop(x_init=processed_scan, x_t=x_full, x_cond=x_uncond, x_uncond=x_uncond, mask=mask)
            completed_scans.append(completed_scan)

            print('Computing metrics for: ', sample_type)
            cd_mean, cd_std = self.compute_metrics(processed_scan.cpu().detach().numpy(), completed_scan, mask.cpu().detach().numpy())
            metrics[sample_type+'_mean'] = cd_mean
            metrics[sample_type+'_std'] = cd_std
        
        x_noise_inpainted = mask_and_return_tensor(input_clean=processed_scan.reshape(1, -1, 3), input_noised=x_feats.reshape(1, -1, 3), mask=mask.squeeze()==0.0)
        return processed_scan, x_noise_inpainted, cond, completed_scans, metrics

    def visualize_step_t(self, x_t, gt_pts, pcd, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        pcd.points = o3d.utility.Vector3dVector(points)
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def in_painting_loop(self, x_init, x_t, x_cond, x_uncond, mask):
        self.scheduler_to_cuda()
        if self.scheduler_type == 'dpm':
            schedule = self.generate_noise_schedule(self.hparams['diff']['s_steps'])
        elif self.scheduler_type == 'repaint':
            schedule = range(len(self.dpm_scheduler.timesteps))

        for t in tqdm.tqdm(schedule):
            t = self.dpm_scheduler.timesteps[t].to(self.device)[None]
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init 

            if self.scheduler_type == 'dpm':
                x_known = x_init + self.dpm_scheduler.add_noise(noise=torch.randn(x_init.shape, device=self.device), original_samples=torch.zeros_like(x_init), timesteps=t)
                x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
                x_t = mask * x_known + (1.0 - mask) * x_t
            elif self.scheduler_type == 'repaint':
                x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise, torch.zeros_like(x_init), mask)['prev_sample']

            x_t = self.points_to_tensor(x_t)
            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond)
            torch.cuda.empty_cache()

        return x_t.F.cpu().detach().numpy()

def mask_and_return_tensor(input_clean, input_noised, mask):
        x_temp = torch.empty_like(input_clean)
        x_temp[:,mask] = input_noised[:,mask]
        x_temp[:,~mask] = input_clean[:,~mask]
        return x_temp

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

def write_pcd_to_file(pcd_points, exp_dir, pcd_name, pcd_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points.reshape(-1, 3))
    pcd.estimate_normals()
    o3d.io.write_point_cloud(f'./results/{exp_dir}/{pcd_name}/{pcd_path}.ply', pcd)

def intercept_3d_volume(target, volume):
    return (target[:, 0] >= volume[0, 0]) & \
    (target[:, 0] <= volume[1, 0]) & \
    (target[:, 1] >= volume[0, 1]) & \
    (target[:, 1] <= volume[1, 1]) & \
    (target[:, 2] >= volume[0, 2]) & \
    (target[:, 2] <= volume[1, 2])

@click.command()
@click.option('--name', '-n', type=str, default='results', help='name of result')
@click.option('--diff', '-d', type=str, default='checkpoints/scan_reconstruction_denoising_1_epoch=19.ckpt', help='path to the scan sequence')
@click.option('--sequence', '-q', type=str, default='08', help='semanti_kitti sequence to sample from')
@click.option('--denoising_steps', '-T', type=int, default=100, help='number of denoising steps (default: 1000)')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weight (default: 6.0)')
@click.option('--sampler_type', '-p', default='repaint', type=str, help='sampler type (default: repaint)')
@click.option('--sample_count', '-c', default=5, type=int, help='number of scenes to sample (default: 5)')
@click.option('--min_points_per_car', '-m', default=1000, type=int, help='min points to consider per car (default: 1000)')
def main(name, diff, sequence, denoising_steps, cond_weight, sampler_type, sample_count, min_points_per_car):
    exp_dir = name +'_'+ f'_T{denoising_steps}_s{cond_weight}'

    in_painter = InPainter(
            diff, denoising_steps, cond_weight, sampler_type
        )

    os.makedirs(f'./results/{exp_dir}', exist_ok=True)

    sequence_paths = glob.glob(f'/datasets_local/semantic_kitti/dataset/sequences/{sequence}/velodyne/**.bin')
    selection = np.random.permutation(len(sequence_paths))[:sample_count]

    metrics = []
    for index in tqdm.tqdm(selection):
        scene_path = sequence_paths[index]
        points = load_pcd(scene_path)

        labels_path = scene_path.replace('velodyne', 'labels').replace('bin', 'label')
        labels = np.fromfile(labels_path, dtype=np.uint32)
        
        instance_ids = (labels >> 16) & 0xFFFF
        labels = labels & 0xFFFF

        cars_instances = instance_ids[(labels==10) | (labels==252)]
        counts = np.unique(cars_instances, return_counts=True)
        valid_car_ids = counts[0][counts[1]>min_points_per_car]
        if valid_car_ids.size == 0:
            continue
        
        scene_label = scene_path[-10:-4]
        os.makedirs(f'./results/{exp_dir}/{scene_label}', exist_ok=True)

        for valid_car_id in valid_car_ids.tolist():
            print(f'Inpainting on scene {scene_path}, car {valid_car_id}')
            mask = ((~(instance_ids == valid_car_id))*1.)[:,None]

            start = time.time()
            original_scan, noise_in_painted_scan, cond, completed_scans, scan_metrics = in_painter.in_paint_scan(points, mask)
            end = time.time()
            print(f'took: {end - start}s')

            os.makedirs(f'./results/{exp_dir}/{scene_label}/{valid_car_id}', exist_ok=True)
            pcd_name = f'{scene_label}/{valid_car_id}'
            
            write_pcd_to_file(original_scan.cpu().detach().numpy(),         exp_dir, pcd_name, pcd_path='original_scan')
            write_pcd_to_file(noise_in_painted_scan.cpu().detach().numpy(), exp_dir, pcd_name, pcd_path='original_scan_inpainted_noise')
            write_pcd_to_file(cond.cpu().detach().numpy(),                  exp_dir, pcd_name, pcd_path='conditioning_scan')
            write_pcd_to_file(completed_scans[0],                           exp_dir, pcd_name, pcd_path='after_inpainting_normal_conditioning')
            write_pcd_to_file(completed_scans[1],                           exp_dir, pcd_name, pcd_path='after_inpainting_no_conditioning')

            metrics.append({'sequence':sequence, 'scene':scene_label, 'car_id':valid_car_id, **scan_metrics})

    with open(f'./results/{exp_dir}/metrics.json', 'w') as fp:
        json.dump(metrics, fp)

if __name__ == '__main__':
    main()
