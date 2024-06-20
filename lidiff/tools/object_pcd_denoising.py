import os
from pytorch_lightning import LightningDataModule
import yaml
from lidiff.models.models_objects_full_diff import DiffusionPoints
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from lidiff.utils.three_d_helpers import build_two_point_clouds
from lidiff.utils.metrics import ChamferDistance
from lidiff.datasets import datasets_objects, datasets_shapenet
from lidiff.utils.data_map import class_mapping
from diffusers import DPMSolverMultistepScheduler
import click

def find_eligible_objects(dataloader, num_to_find=1, object_class='vehicle.car', min_points=None):
    targets = []

    for index, item in enumerate(dataloader):
        num_lidar_points = item['num_points'][0]
        class_index = torch.argmax(item['class'])
        item['index'] = index
        if class_mapping[object_class] != class_index:
            continue

        if num_lidar_points > min_points:
            targets.append(item)
        
        if len(targets) >= num_to_find:
            break

    return targets

def visualize_step_t(x_t, pcd):
    points = x_t.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def p_sample_loop(model: DiffusionPoints, x_t, x_cond, x_uncond, x_class, batch_indices, viz_path=None):
    model.scheduler_to_cuda()
    generate_viz = viz_path != None
    if generate_viz:
        viz_pcd = o3d.geometry.PointCloud()
        os.makedirs(f'{viz_path}/step_visualizations', exist_ok=True)

    for t in tqdm(range(len(model.dpm_scheduler.timesteps))):
        t = model.dpm_scheduler.timesteps[t].cuda()[None]

        with torch.no_grad():
            noise_t = model.classfree_forward(x_t, x_cond, x_uncond, t, x_class).squeeze(0)
            torch.cuda.empty_cache()
        input_noise = x_t.F

        x_t = model.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
        
        x_t = model.points_to_tensor(x_t, batch_indices)

        if generate_viz:
            viz = visualize_step_t(x_t.F.clone(), viz_pcd)
            print(f'Saving Visualization of Step {t}')
            o3d.io.write_point_cloud(f'{viz_path}/step_visualizations/object_gen_viz_step_{t[0]}.ply', viz)

        torch.cuda.empty_cache()

    return x_t
        
def denoise_object_from_pcd(model: DiffusionPoints, x_object, x_center, x_size, x_orientation, x_class, num_diff_samples, viz_path=None):
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True

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

    x_init = x_object.clone().cuda()    
    batch_indices = torch.zeros(x_init.shape[0]).long().cuda()

    x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1).cuda()
    x_uncond = torch.zeros_like(x_cond).cuda()
    x_class = x_class.cuda()

    local_chamfer = ChamferDistance()
    x_gen_evals = []
    for i in tqdm(range(num_diff_samples)):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        x_feats = torch.randn(x_init.shape, device=model.device)
        x_t = model.points_to_tensor(x_feats, batch_indices)
        x_gen_eval = p_sample_loop(model, x_t, x_cond, x_uncond, x_class, batch_indices, viz_path=viz_path)
        x_gen_evals.append(x_gen_eval.F)    
        pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=x_gen_eval.F, object_pcd=x_init)
        local_chamfer.update(pcd_gt, pcd_pred)
        print(f"Seed {i} CD: {local_chamfer.last_cd()}")

    best_index = local_chamfer.best_index()
    print(f"Best seed: {best_index}")
    x_gen_eval = x_gen_evals[best_index]

    return x_gen_eval.cpu().detach().numpy(), x_object.cpu().detach().numpy()

def extract_object_info(object_info):
    x_object = object_info['pcd_object']
    x_center = object_info['center']
    x_size = object_info['size']
    x_orientation = object_info['orientation']
    x_class = object_info['class']

    return x_object, x_center, x_size, x_orientation, x_class

def find_pcd_and_test_on_object(dir_path, model, do_viz, objects):
    for _, object_info in enumerate(objects):
        x_gen, x_orig = denoise_object_from_pcd(
            model,
            *extract_object_info(object_info),
            10,
            viz_path=f'{dir_path}' if do_viz else None
        )
        np.savetxt(f'{dir_path}/generated_{object_info["index"]}.txt', x_gen)
        np.savetxt(f'{dir_path}/original_{object_info["index"]}.txt', x_orig)

def find_pcd_and_interpolate_condition(dir_path, conditions, model, objects, do_viz):
    for object_info in objects:
        print(f'Generating using car info {object_info["index"]}')
        pcd, center_cyl, size, yaw, x_class = extract_object_info(object_info)
        def do_gen(condition, index, center_cyl=center_cyl, size=size, yaw=yaw):
                x_gen, x_orig = denoise_object_from_pcd(
                    model=model,
                    x_object=pcd,
                    x_center=center_cyl,
                    x_size=size, 
                    x_orientation=yaw,
                    x_class=x_class,
                    num_diff_samples=1,
                    viz_path=f'{dir_path}' if do_viz else None
                )
                np.savetxt(f'{dir_path}/object_{object_info["index"]}_{condition}_interp_{index}.txt', x_gen)
                np.savetxt(f'{dir_path}/object_{object_info["index"]}_orig.txt', x_orig)

        for condition in conditions:
            if condition == 'angle':
                orig = yaw
                angles = np.linspace(start=orig, stop=orig*-1, num=5)
                print(f'Interpolating Yaw')
                for index, angle in enumerate(angles):
                    do_gen(condition, index=index, yaw=torch.from_numpy(angle))
            if condition == 'center':
                start = center_cyl[:, 0]
                linspace_ring = np.linspace(start=start, stop=start*-1, num=3)
                start = center_cyl[:, 1]
                linspace_dist = np.linspace(start=start, stop=start*2, num=3)
                start = center_cyl[:, 2]
                linspace_vertical_dist = np.linspace(start=start-1, stop=start+1, num=5)
                print(f'Interpolating Cylindrical Angle')
                for index, ring in enumerate(linspace_ring):
                    new_cyl = center_cyl.clone()
                    new_cyl[:,0] = ring.item()
                    do_gen(condition+'_angle', index=index, center_cyl=new_cyl)
                print(f'Interpolating Cylindrical Distance')
                for index, dist in enumerate(linspace_dist):
                    new_cyl = center_cyl.clone()
                    new_cyl[:,1] = dist.item()
                    do_gen(condition+'_distance', index=index, center_cyl=new_cyl)
                print(f'Interpolating Vertical Distance')
                for index, dist in enumerate(linspace_vertical_dist):
                    new_cyl = center_cyl.clone()
                    new_cyl[:,2] = dist.item()
                    do_gen(condition+'_vertical_distance', index=index, center_cyl=new_cyl)
            
@click.command()
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)'
            )
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt).',
              default=None
            )
@click.option('--output_path',
              '-o',
              type=str,
              help='path to save the generated point clouds',
              default='lidiff/random_pcds/generated_pcd'
            )
@click.option('--name',
              '-n',
              type=str,
              help='folder in which generated point clouds will be saved.',
              default=None
            )
@click.option('--task',
              '-t',
              type=str,
              help='Task to run. options are recreate or interpolate',
              default=None)
@click.option('--class_name',
              '-cls',
              type=str,
              help='Label of class to generate.',
              default='vehicle.car'
            )
@click.option('--split',
              '-s',
              type=str,
              help='Which split to take the conditioning information from.',
              default='train'
            )
@click.option('--min_points',
              '-m',
              type=int,
              help='Minimum number of points per cloud.',
              default=100
            )
@click.option('--do_viz',
              '-v',
              type=bool,
              help='Generate step visualizations (every step). True or False.',
              default=False
            )
@click.option('--examples_to_generate',
              '-e',
              type=int,
              help='Number of examples to generate',
              default=1
            )
def main(config, weights, output_path, name, task, class_name, split, min_points, do_viz, examples_to_generate):
    dir_path = f'{output_path}/{name}'
    os.makedirs(dir_path, exist_ok=True)
    cfg = yaml.safe_load(open(config))
    cfg['diff']['s_steps'] = 1000
    cfg['diff']['uncond_w'] = 6.
    cfg['train']['batch_size'] = 1
    model = DiffusionPoints.load_from_checkpoint(weights, hparams=cfg).cuda()
    model.eval()

    dataloader_maps = [datasets_shapenet.dataloaders, datasets_objects.dataloaders]
    for map in dataloader_maps:
        if cfg['data']['dataloader'] in map:
           module: LightningDataModule = map[cfg['data']['dataloader']](cfg)
           break

    dataloader = module.train_dataloader() if split == 'train' else module.val_dataloader()
    objects = find_eligible_objects(dataloader, num_to_find=examples_to_generate, object_class=class_name, min_points=min_points)
    if task == 'recreate':
        find_pcd_and_test_on_object(dir_path=dir_path, model=model, objects=objects, do_viz=do_viz)
    if task == 'interpolate':
        find_pcd_and_interpolate_condition(dir_path=dir_path, conditions=['angle','center',], model=model, do_viz=do_viz, objects=objects)

if __name__ == "__main__":
    main()
