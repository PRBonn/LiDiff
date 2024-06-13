from ctypes import py_object
from decimal import localcontext
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
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import LightningDataModule
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from lidiff.utils.three_d_helpers import build_two_point_clouds
from diffusers import DPMSolverMultistepScheduler
from random import shuffle

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

        self.model = minknet.MinkUNetDiff(
            in_channels=3, 
            out_channels=self.hparams['model']['out_dim'], 
            embeddings_type=self.hparams['model']['embeddings']
        )

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

    def visualize_step_t(self, x_t, gt_pts, pcd):
        points = x_t.detach().cpu().numpy()
        points = np.concatenate((points, gt_pts.detach().cpu().numpy()), axis=0)

        pcd.points = o3d.utility.Vector3dVector(points)
       
        colors = np.ones((len(points), 3)) * .5
        colors[:len(gt_pts[0])] = [1.,.3,.3]
        colors[-len(gt_pts[0]):] = [.3,1.,.3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def p_sample_loop(self, x_t, x_cond, x_uncond, batch_indices, num_points):
        self.scheduler_to_cuda()

        for t in tqdm(range(len(self.dpm_scheduler.timesteps))):
            random_ints = torch.ones(num_points.shape[0]).cuda().long() * self.dpm_scheduler.timesteps[t].cuda()
            t = random_ints[batch_indices]            
            
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t).squeeze(1)
            input_noise = x_t.F

            x_t = self.dpm_scheduler.step(noise_t, t[0], input_noise)['prev_sample']
            
            x_t = self.points_to_tensor(x_t, batch_indices)

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
        t_sample = self.q_sample(batch['pcd_object'], t, noise)

        # replace the original points with the noise sampled
        x_object_noised = self.points_to_tensor(t_sample, batch['batch_indices'])

        # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or batch['pcd_object'].shape[0] == 1:
            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']
            x_class = batch['class']
        else:
            x_center = torch.zeros_like(batch['center'])
            x_size = torch.zeros_like(batch['size'])
            x_orientation = torch.zeros_like(batch['orientation'])
            x_class = torch.zeros_like(batch['class'])
        
        if self.hparams['model']['embeddings'] == 'cyclical':
            x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
        else:
            x_cond = torch.hstack((x_center, x_size, x_orientation))

        x_sparse = x_object_noised.sparse()
        denoise_t = self.forward(x_object_noised, x_sparse, t, x_cond).squeeze(1)
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
        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']

            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']

            if self.hparams['model']['embeddings'] == 'cyclical':
                x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))
            x_uncond = torch.zeros_like(x_cond)

            x_t = torch.randn(x_object.shape, device=self.device)
            x_t = self.points_to_tensor(x_t, batch['batch_indices'])
            x_gen_eval = self.p_sample_loop(x_t, x_cond, x_uncond, batch['batch_indices'], batch['num_points']).F

            curr_index = 0
            cd_mean_as_pct_of_box = []
            
            for pcd_index in range(batch['num_points'].shape[0]):
                max_index = int(curr_index + batch['num_points'][pcd_index].item())
                object_pcd = x_object[curr_index:max_index]
                genrtd_pcd = x_gen_eval[curr_index:max_index]

                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_pcd, object_pcd=object_pcd)

                self.chamfer_distance.update(pcd_gt, pcd_pred)
                self.precision_recall.update(pcd_gt, pcd_pred)

                last_cd = self.chamfer_distance.last_cd()
                box = batch['size'][pcd_index].cpu()
                cd_mean_as_pct_of_box.append((last_cd / box.mean())*100.)
                curr_index = max_index

        cd_mean, cd_std = self.chamfer_distance.compute()
        cd_mean_as_pct_of_box = np.mean(cd_mean_as_pct_of_box)
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}\tAs % of Box: {cd_mean_as_pct_of_box}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/cd_mean_as_pct_of_box', cd_mean_as_pct_of_box, on_step=True)
        torch.cuda.empty_cache()

        return {'val/cd_mean': cd_mean, 'val/cd_std': cd_std, 'val/cd_as_pct_of_box':cd_mean_as_pct_of_box}
    
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
        self.model.eval()
        
        viz_pcd = o3d.geometry.PointCloud()
        makedirs(f'{self.logger.log_dir}/generated_pcd/visualizations', exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            x_object = batch['pcd_object']

            x_center = batch['center']
            x_size = batch['size']
            x_orientation = batch['orientation']

            if self.hparams['model']['embeddings'] == 'cyclical':
                x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1)
            else:
                x_cond = torch.hstack((x_center, x_size, x_orientation))
            x_uncond = torch.zeros_like(x_cond)

            x_gen_evals = []
            num_val_samples = self.hparams['diff']['num_val_samples']
            for i in range(num_val_samples):
                np.random.seed(i)
                torch.manual_seed(i)
                torch.cuda.manual_seed(i)
                noise = torch.randn(x_object.shape, device=self.device)
                x_t = self.points_to_tensor(noise, batch['batch_indices'])
                x_gen_eval = self.p_sample_loop(x_t, x_cond, x_uncond, batch['batch_indices'], batch['num_points'])
                x_gen_evals.append(x_gen_eval.F)

            curr_index = 0
            cd_mean_as_pct_of_box = []
            for pcd_index in range(batch['num_points'].shape[0]):
                max_index = int(curr_index + batch['num_points'][pcd_index].item())
                object_pcd = x_object[curr_index:max_index]
                
                local_chamfer = ChamferDistance()
                for generated_pcds in x_gen_evals:
                    genrtd_pcd = generated_pcds[curr_index:max_index]

                    pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_pcd, object_pcd=object_pcd)

                    local_chamfer.update(pcd_gt, pcd_pred)

                best_index = local_chamfer.best_index()
                genrtd_pcd = x_gen_evals[best_index][curr_index:max_index]

                pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=genrtd_pcd, object_pcd=object_pcd)

                self.chamfer_distance.update(pcd_gt, pcd_pred)

                last_cd = self.chamfer_distance.last_cd()
                box = batch['size'][pcd_index].cpu()
                cd_mean_as_pct_of_box.append((last_cd / box.mean())*100.)
                curr_index = max_index
                if pcd_index == 0:
                    visualization_1 = self.visualize_step_t(genrtd_pcd, object_pcd, viz_pcd)
                    o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/visualizations/batch_{batch_idx}_object_{pcd_index}_seed_{best_index}_best.ply', visualization_1)
                    random_choices = [i for i in range(num_val_samples) if i != best_index]
                    shuffle(random_choices)
                    for i in random_choices[0:2]:
                        genrtd_pcd_2 = x_gen_evals[i][curr_index:max_index]
                        visualization_2 = self.visualize_step_t(genrtd_pcd_2, object_pcd, viz_pcd)
                        o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/visualizations/batch_{batch_idx}_object_{pcd_index}_seed_{i}.ply', visualization_2)

        cd_mean, cd_std = self.chamfer_distance.compute()
        cd_mean_as_pct_of_box = np.mean(cd_mean_as_pct_of_box)
        print(f'CD Mean: {cd_mean}\tCD Std: {cd_std}\tAs % of Box: {cd_mean_as_pct_of_box}')

        self.log('test/cd_mean', cd_mean, on_step=True)
        self.log('test/cd_std', cd_std, on_step=True)
        self.log('test/cd_mean_as_pct_of_box', cd_mean_as_pct_of_box, on_step=True)
        torch.cuda.empty_cache()

        return {'test/cd_mean': cd_mean, 'test/cd_std': cd_std, 'test/cd_mean_as_pct_of_box':cd_mean_as_pct_of_box,}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        scheduler = {
            # 'scheduler': scheduler, # lr * 0.5
            'interval': 'epoch', # interval is epoch-wise
            'frequency': 5, # after 5 epochs
        }

        return [optimizer]

#######################################
# Modules
#######################################
