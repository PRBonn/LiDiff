from operator import xor
import os
import yaml
from lidiff.models.models_objects_full_diff import DiffusionPoints
import torch
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix, points_in_box
from nuscenes.utils.data_classes import Box, Quaternion
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits
import open3d as o3d
from tqdm import tqdm
from lidiff.utils.three_d_helpers import cartesian_to_cylindrical, cartesian_to_spherical, extract_yaw_angle, build_two_point_clouds
from lidiff.utils.metrics import ChamferDistance
from diffusers import DPMSolverMultistepScheduler
import sys

def get_min_points_from_class(object_class):
    return {
        'vehicle.car':300,
        'vehicle.bicycle':100
    }[object_class]

def find_eligible_objects(num_to_find=1, object_class='vehicle.car'):
    dataroot = '/datasets_local/nuscenes'
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)

    train_split = set(splits.train)
    targets = []
    found_target = False
    min_points = get_min_points_from_class(object_class)
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        sample_data_lidar_token = sample['data']['LIDAR_TOP']
        scene_name = nusc.get('scene', scene_token)['name']
        if scene_name in train_split:
            continue

        lidar_data = nusc.get('sample_data', sample_data_lidar_token)
        lidar_filepath = os.path.join(dataroot, lidar_data['filename'])
        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
        objects = nusc.get_sample_data(sample_data_lidar_token)[1]

        for object in objects:
            if object_class in object.name:
                annotation = nusc.get('sample_annotation', object.token)
                num_lidar_points = annotation['num_lidar_pts']
                points = points_in_box(object, lidar_pointcloud.points[:3, :])
                object_info = {
                    'sample_token': object.token,
                    'lidar_filepath': lidar_filepath,
                    'center': object.center.tolist(),
                    'wlh': object.wlh.tolist(),
                    'rotation_real': object.orientation.real.tolist(),
                    'rotation_imaginary': object.orientation.imaginary.tolist(),
                    'points': points.tolist(),
                    'num_lidar_points': num_lidar_points
                }
                
                if num_lidar_points > min_points:
                    targets.append(object_info)
                    if len(targets) >= num_to_find:
                        found_target = True
                        break
        if found_target:
            break
    return targets

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 5))[:, :3]

    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

def visualize_step_t(x_t, pcd):
    points = x_t.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def p_sample_loop(model: DiffusionPoints, x_init, x_t, x_cond, x_uncond, batch_indices, generate_viz=False):
    model.scheduler_to_cuda()
    if generate_viz:
        viz_pcd = o3d.geometry.PointCloud()
        os.makedirs(f'lidiff/random_pcds/generated_pcd/step_visualizations', exist_ok=True)

    for t in tqdm(range(len(model.dpm_scheduler.timesteps))):
        t = model.dpm_scheduler.timesteps[t].cuda()[None]

        with torch.no_grad():
            noise_t = model.classfree_forward(x_t, x_cond, x_uncond, t).squeeze(0)
            torch.cuda.empty_cache()
        input_noise = x_t.F

        x_t = model.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
        
        x_t = model.points_to_tensor(x_t, batch_indices)

        if generate_viz:
            viz = visualize_step_t(x_t.F.clone(), viz_pcd)
            print(f'Saving Visualization of Step {t}')
            o3d.io.write_point_cloud(f'lidiff/random_pcds/generated_pcd/step_visualizations/full_diff_cylindrical_coord_step_{t[0]}.ply', viz)

        torch.cuda.empty_cache()

    return x_t
        
def denoise_object_from_pcd(model: DiffusionPoints, car_points, center_cyl, wlh, angle, num_diff_samples):
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

    x_size = torch.Tensor(wlh).float().unsqueeze(0)
    x_orientation = torch.ones((1,1)) * angle
    x_object = torch.from_numpy(car_points)
    x_center = torch.from_numpy(center_cyl).float()

    x_init = x_object.clone().cuda()    
    batch_indices = torch.zeros(x_init.shape[0]).long().cuda()

    x_cond = torch.cat((torch.hstack((x_center[:,0][:, None], x_orientation)), torch.hstack((x_center[:,1:], x_size))),-1).cuda()
    x_uncond = torch.zeros_like(x_cond).cuda()

    local_chamfer = ChamferDistance()
    x_gen_evals = []
    for i in tqdm(range(num_diff_samples)):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        x_feats = torch.randn(x_init.shape, device=model.device)
        x_t = model.points_to_tensor(x_feats, batch_indices)
        x_gen_eval = p_sample_loop(model, x_init, x_t, x_cond, x_uncond, batch_indices, generate_viz=False)
        x_gen_evals.append(x_gen_eval.F)    
        pcd_pred, pcd_gt = build_two_point_clouds(genrtd_pcd=x_gen_eval.F, object_pcd=x_init)
        local_chamfer.update(pcd_gt, pcd_pred)
        print(f"Seed {i} CD: {local_chamfer.last_cd()}")

    best_index = local_chamfer.best_index()
    print(f"Best seed: {best_index}")
    x_gen_eval = x_gen_evals[best_index]

    return x_gen_eval.cpu().detach().numpy(), x_object.cpu().detach().numpy()

def extract_object_info(object_info):
    rotation_real = np.array(object_info['rotation_real'])
    rotation_imaginary = np.array(object_info['rotation_imaginary'])
    orientation = Quaternion(real=rotation_real, imaginary=rotation_imaginary)
    size = np.array(object_info['wlh'])
    center = np.array(object_info['center'])
    pcd = load_pcd(object_info['lidar_filepath'])[object_info['points']] - center
    center_cyl = cartesian_to_cylindrical(center[None, :])
    return pcd, center_cyl, size, extract_yaw_angle(orientation)

def find_pcd_and_test_on_object(output_path, name, model, class_name):
    object_info = find_eligible_objects(object_class=class_name)[0]
    pcd, center_cyl, size, yaw = extract_object_info(object_info)

    x_gen, x_orig = denoise_object_from_pcd(
        model,
        pcd, 
        center_cyl, 
        size, 
        yaw,
        5,
    )
    np.savetxt(f'{output_path}/{name}/generated.txt', x_gen)
    np.savetxt(f'{output_path}/{name}/orig.txt', x_orig)

def find_pcd_and_interpolate_condition(output_path, name, conditions, num_to_find, model, class_name):
    object_infos = find_eligible_objects(num_to_find=num_to_find, object_class=class_name)
    for object_info in object_infos:
        print(f'Generating using car info {object_info["sample_token"]}')
        pcd, center_cyl, size, yaw = extract_object_info(object_info)
        def do_gen(condition, index, center_cyl=center_cyl, size=size, yaw=yaw):
                x_gen, x_orig = denoise_object_from_pcd(
                    model,
                    pcd,
                    center_cyl,
                    size, 
                    yaw,
                    1,
                )
                np.savetxt(f'{output_path}/{name}/{object_info["sample_token"]}_{condition}_interp_{index}.txt', x_gen)
                np.savetxt(f'{output_path}/{name}/{object_info["sample_token"]}_orig.txt', x_orig)

        for condition in conditions:
            if condition == 'angle':
                orig = yaw
                angles = np.linspace(start=orig, stop=orig*-1, num=5)
                print(f'Interpolating Yaw')
                for index, angle in enumerate(angles):
                    do_gen(condition, index=index, yaw=angle)
            if condition == 'center':
                start = center_cyl[:, 0]
                linspace_ring = np.linspace(start=start, stop=start*-1, num=5)
                start = center_cyl[:, 1]
                linspace_dist = np.linspace(start=start, stop=start*2, num=3)
                start = center_cyl[:, 2]
                linspace_vertical_dist = np.linspace(start=start-1, stop=start+1, num=5)
                print(f'Interpolating Cylindrical Angle')
                for index, ring in enumerate(linspace_ring):
                    new_cyl = center_cyl.copy()
                    new_cyl[:,0] = ring
                    do_gen(condition+'_angle', index=index, center_cyl=new_cyl)
                print(f'Interpolating Cylindrical Distance')
                for index, dist in enumerate(linspace_dist):
                    new_cyl = center_cyl.copy()
                    new_cyl[:,1] = dist
                    do_gen(condition+'_distance', index=index, center_cyl=new_cyl)
                print(f'Interpolating Vertical Distance')
                for index, dist in enumerate(linspace_vertical_dist):
                    new_cyl = center_cyl.copy()
                    new_cyl[:,2] = dist
                    do_gen(condition+'_vertical_distance', index=index, center_cyl=new_cyl)
            

def main(task, class_name):
    output_path, name = 'lidiff/random_pcds/generated_pcd', 'bike_gen_1'
    os.makedirs(f'{output_path}/{name}/', exist_ok=True)
    config = 'lidiff/config/object_generation/config_bike_gen.yaml'
    weights = 'lidiff/checkpoints/bike_gen_1_epoch=99.ckpt'
    cfg = yaml.safe_load(open(config))
    cfg['diff']['s_steps'] = 1000
    cfg['diff']['uncond_w'] = 6.
    model = DiffusionPoints.load_from_checkpoint(weights, hparams=cfg).cuda()
    model.eval()

    if task == 'recreate':
        find_pcd_and_test_on_object(output_path, name, model, class_name)
    if task == 'interpolate':
        find_pcd_and_interpolate_condition(output_path, name, conditions=['angle','center'], num_to_find=5, model=model, class_name=class_name)
if __name__=='__main__':
    task = sys.argv[1]
    class_name = sys.argv[2]
    main(task, class_name)
