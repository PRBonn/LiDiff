import torch
import torch.nn as nn
import torch.nn.functional as F
import lidiff.models.minkunet_objects as minknet
import numpy as np
import MinkowskiEngine as ME
import open3d as o3d
from lidiff.utils.scheduling import beta_func
from tqdm import tqdm
from os import makedirs, path
import time

from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        # alphas and betas
        if self.hparams['diff']['beta_func'] == 'cosine':
            self.betas = beta_func[self.hparams['diff']['beta_func']](self.hparams['diff']['t_steps'])
        else:
            self.betas = beta_func[self.hparams['diff']['beta_func']](
                    self.hparams['diff']['t_steps'],
                    self.hparams['diff']['beta_start'],
                    self.hparams['diff']['beta_end'],
            )

        self.t_steps = self.hparams['diff']['t_steps']
        self.s_steps = self.hparams['diff']['s_steps']
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.tensor(
            np.cumprod(self.alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
        )

        self.alphas_cumprod_prev = torch.tensor(
            np.append(1., self.alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
        )

        self.betas = torch.tensor(self.betas, device=torch.device('cuda'))
        self.alphas = torch.tensor(self.alphas, device=torch.device('cuda'))

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod) 
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.sqrt_posterior_variance = torch.sqrt(self.posterior_variance)
        self.posterior_log_var = torch.log(
            torch.max(self.posterior_variance, 1e-20 * torch.ones_like(self.posterior_variance))
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # for fast sampling
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.t_steps,
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.s_steps)
        self.scheduler_to_cuda()

        self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.chamfer_distance = ChamferDistance()
        self.precision_recall = PrecisionRecall(self.hparams['data']['resolution'],2*self.hparams['data']['resolution'],100)

        self.w_uncond = self.hparams['train']['uncond_w']
        self.visualize = self.hparams['diff']['visualize']

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def q_sample(self, x, t, noise):
        return self.sqrt_alphas_cumprod[t][:,None].cuda() * x + \
                self.sqrt_one_minus_alphas_cumprod[t][:,None].cuda() * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, t, x_cond)            
        x_uncond = self.forward(x_t, x_t_sparse, t,  x_uncond)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def visualize_step_t(self, x_t, gt_pts, pcd, pcd_mean, pcd_std, pidx=0):
        points = x_t.F.detach().cpu().numpy()
        points = points.reshape(gt_pts.shape[0],-1,3)
        obj_mean = pcd_mean[pidx][0].detach().cpu().numpy()
        points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

        dist_pts = np.sqrt(np.sum((points - obj_mean)**2, axis=-1))
        dist_idx = dist_pts < self.hparams['data']['max_range']

        full_pcd = len(points) - len(gt_pts[pidx])
        print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

        pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])
        return pcd

    def reset_partial_pcd(self, x_part, x_uncond, x_mean, x_std):
        x_part = self.points_to_tensor(x_part.F.reshape(x_mean.shape[0],-1,3).detach(), x_mean, x_std)
        x_uncond = self.points_to_tensor(
                torch.zeros_like(x_part.F.reshape(x_mean.shape[0],-1,3)), torch.zeros_like(x_mean), torch.zeros_like(x_std)
        )

        return x_part, x_uncond

    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond, batch_indices, num_points, generate_viz=False):
        self.scheduler_to_cuda()
        if generate_viz:
            viz_pcd = o3d.geometry.PointCloud()
            makedirs(f'{self.logger.log_dir}/generated_pcd/step_visualizations', exist_ok=True)

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            random_ints = torch.ones(num_points.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            t = random_ints[batch_indices]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t).squeeze(1)
            input_noise = x_t.F - x_init

            x_t = x_init + self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            
            x_t = self.points_to_tensor(x_t, batch_indices)

            if generate_viz:
                viz = self.visualize_step_t(x_t, x_init.cpu().detach().numpy(), viz_pcd)
                print(f'Saving Visualization of Step {t}')
                o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/step_visualizations/step_{t[0]}.ply', viz)

            torch.cuda.empty_cache()

        

        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def forward(self, x_object, x_sparse, t, conditions):
        out = self.model(x_object, x_sparse, t, conditions)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, x_feats, batched_indices):
        x_coord = x_feats.clone()
        x_coord = torch.round(x_feats / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=x_feats.float(),
            coordinates=torch.concat((batched_indices[:, None], x_coord), dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def training_step(self, batch:dict, batch_idx):
        # initial random noise
        torch.cuda.empty_cache()
        noise = torch.randn(batch['pcd_object'].shape, device=self.device)
        
        # sample step t
        random_ints = torch.randint(0, self.t_steps, size=(batch['num_points'].shape[0],)).cuda()
        t = random_ints[batch['batch_indices']]
        # sample q at step t
        # we sample noise towards zero to then add to each point the noise (without normalizing the pcd)
        t_sample = batch['pcd_object'] + self.q_sample(torch.zeros_like(batch['pcd_object']), t, noise)

        # replace the original points with the noise sampled
        x_object_noised = self.points_to_tensor(t_sample, batch['batch_indices'])

        # for classifier-free guidance switch between conditional and unconditional training NOTE THIS NEEDS TO CHANGE FOR BATCH SIZE 1 TRAINING
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_object'].shape[0] == 1:
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
        else:
            x_center = torch.zeros_like(batch['center'])
            x_size = torch.zeros_like(batch['size'])
            x_orientation = torch.zeros_like(batch['orientation'])
        
        conditions = torch.hstack((x_center, x_size, x_orientation))
        x_sparse = x_object_noised.sparse()
        denoise_t = self.forward(x_object_noised, x_sparse, t, conditions).squeeze(1)
        loss_mse = self.p_losses(denoise_t, noise)
        loss_mean = (denoise_t.mean())**2
        loss_std = (denoise_t.std() - 1.)**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)

        std_noise = (denoise_t - noise)**2
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        self.log('train/var', std_noise.var())
        self.log('train/std', std_noise.std())
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.model.eval()
        with torch.no_grad():
            # for inference we get the partial pcd and sample the noise around the partial
            x_init = batch['pcd_object']
            x_feats = x_init + torch.randn(x_init.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['batch_indices'])

            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']

            x_cond = torch.hstack((x_center, x_size, x_orientation))
            x_uncond = torch.zeros_like(x_cond)

            x_gen_eval = self.p_sample_loop(x_init, x_full, x_cond, x_uncond, batch['batch_indices'], batch['num_points'])
            x_gen_eval = x_gen_eval.F

            pcd_pred = o3d.geometry.PointCloud()
            c_pred = x_gen_eval.cpu().detach().numpy()
            pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

            pcd_gt = o3d.geometry.PointCloud()
            g_pred = batch['pcd_object'].cpu().detach().numpy()
            pcd_gt.points = o3d.utility.Vector3dVector(g_pred)

            self.chamfer_distance.update(pcd_gt, pcd_pred)
            self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()

        self.log('val/cd_mean', cd_mean, on_step=True)
        self.log('val/cd_std', cd_std, on_step=True)
        self.log('val/precision', pr, on_step=True)
        self.log('val/recall', re, on_step=True)
        self.log('val/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/precision': pr, 'val/recall': re, 'val/fscore': f1}
    
    def valid_paths(self, filenames):
        output_paths = []
        skip = []

        for fname in filenames:
            seq_dir =  f'{self.logger.log_dir}/generated_pcd/{fname.split("/")[-3]}'
            ply_name = f'{fname.split("/")[-1].split(".")[0]}.ply'

            skip.append(path.isfile(f'{seq_dir}/{ply_name}'))
            makedirs(seq_dir, exist_ok=True)
            output_paths.append(f'{seq_dir}/{ply_name}')

        return np.all(skip), output_paths

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        with torch.no_grad():
            skip, output_paths = self.valid_paths(batch['filename'])

            if skip:
                print(f'Skipping generation from {output_paths[0]} to {output_paths[-1]}') 
                return {'test/cd_mean': 0., 'test/cd_std': 0., 'test/precision': 0., 'test/recall': 0., 'test/fscore': 0.}

            scan = batch['pcd_full']
            x_feats = scan + torch.randn(scan.shape, device=self.device)
            x_full = self.points_to_tensor(x_feats, batch['mean'], batch['std'])
            x_part = self.points_to_tensor(batch['pcd_part'], batch['mean'], batch['std'])
            x_uncond = self.points_to_tensor(
                torch.zeros_like(batch['pcd_part']), torch.zeros_like(batch['mean']), torch.zeros_like(batch['std'])
            )

            x_gen_eval = self.p_sample_loop(scan, x_full, x_part, x_uncond, batch['mean'], batch['std'], self.visualize)
            x_gen_eval = x_gen_eval.F.reshape((scan.shape[0],-1,3))

            for i in range(len(batch['pcd_full'])):
                pcd_pred = o3d.geometry.PointCloud()
                c_pred = x_gen_eval[i].cpu().detach().numpy()
                dist_pts = np.sqrt(np.sum((c_pred)**2, axis=-1))
                dist_idx = dist_pts < self.hparams['data']['max_range']
                points = c_pred[dist_idx]
                max_z = scan[i][...,2].max().item()
                min_z = (scan[i][...,2].mean() - 2 * scan[i][...,2].std()).item()
                pcd_pred.points = o3d.utility.Vector3dVector(points[(points[:,2] < max_z) & (points[:,2] > min_z)])
                pcd_pred.paint_uniform_color([1.0, 0.,0.])

                pcd_gt = o3d.geometry.PointCloud()
                g_pred = batch['pcd_full'][i].cpu().detach().numpy()
                pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
                pcd_gt.paint_uniform_color([0., 1.,0.])
                
                print(f'Saving {output_paths[i]}')
                o3d.io.write_point_cloud(f'{output_paths[i]}', pcd_pred)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

        cd_mean, cd_std = self.chamfer_distance.compute()
        pr, re, f1 = self.precision_recall.compute_auc()
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}')
        print(f'Precision: {pr}\tRecall: {re}\tF-Score: {f1}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/precision', pr, on_step=True)
        self.log('test/recall', re, on_step=True)
        self.log('test/fscore', f1, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/precision': pr, 'test/recall': re, 'test/fscore': f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        scheduler = {
            'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer], [scheduler]

#######################################
# Modules
#######################################
