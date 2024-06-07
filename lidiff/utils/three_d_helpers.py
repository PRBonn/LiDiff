import numpy as np
from pyquaternion import Quaternion
import open3d as o3d

def extract_yaw_angle(quaternion: Quaternion):
    yaw = quaternion.yaw_pitch_roll[0]
    return yaw

def cartesian_to_spherical(xyz):
    ptsnew = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def cartesian_to_cylindrical(xyz):
    x,y,z = np.split(xyz, 3, axis=1)
    dist = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    # Order makes it easier to break cyclical part off
    return np.concatenate((angle, dist, z), axis=1)

def build_two_point_clouds(genrtd_pcd, object_pcd):
    pcd_pred = o3d.geometry.PointCloud()
    c_pred = genrtd_pcd.cpu().detach().numpy()
    pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

    pcd_gt = o3d.geometry.PointCloud()
    g_pred = object_pcd.cpu().detach().numpy()
    pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
    return pcd_pred, pcd_gt