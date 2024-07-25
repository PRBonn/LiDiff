import os
import torch
import numpy as np
import open3d as o3d
from natsort import natsorted
import click
import tqdm
import MinkowskiEngine as ME

def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib

def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

@click.command()
@click.option('--path', '-p', type=str, help='path to the scan sequence')
@click.option('--voxel_size', '-v', type=float, default=0.1, help='voxel size')
@click.option('--cpu', '-c', is_flag=True, help='Use CPU')
def main(path, voxel_size, cpu):
    device_label = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_label = 'cpu' if cpu else device_label
    device = torch.device(device_label)
    for seq in ['00','01','02','03','04','05','06','07','08','09','10']:
        map_points = torch.empty((0,3)).to(device)

        poses = load_poses(os.path.join(path, seq, 'calib.txt'), os.path.join(path, seq, 'poses.txt'))
        for pose, pcd_path in tqdm.tqdm(list(zip(poses, natsorted(os.listdir(os.path.join(path, seq, 'velodyne')))))):
            pose = torch.from_numpy(pose).float().to(device)
            pcd_file = os.path.join(path, seq, 'velodyne', pcd_path)
            points = torch.from_numpy(np.fromfile(pcd_file, dtype=np.float32)).to(device)
            points = points.reshape(-1,4)

            label_file = pcd_file.replace('velodyne', 'labels').replace('.bin', '.label')
            l_set = np.fromfile(label_file, dtype=np.uint32)
            l_set = l_set.reshape((-1))
            l_set = l_set & 0xFFFF

            # remove moving points
            static_idx = (l_set < 252) & (l_set > 1)
            points = points[static_idx]

            # remove flying artifacts
            dist = torch.pow(points, 2)
            dist = torch.sqrt(dist.sum(-1))
            points = points[dist > 3.5]

            points[:,-1] = 1.
            points = points @ pose.T

            map_points = torch.cat((map_points, points[:,:3]), axis=0)
            _, mapping = ME.utils.sparse_quantize(coordinates=map_points / voxel_size, return_index=True, device=device_label)
            map_points = map_points[mapping]


        print(f'saving map for sequence {seq}')
        np.save(os.path.join(path, seq, 'map_clean.npy'), map_points.cpu().numpy())


if __name__ == '__main__':
    main()
