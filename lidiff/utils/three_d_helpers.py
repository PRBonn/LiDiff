import numpy as np
from pyquaternion import Quaternion
import open3d as o3d

def extract_yaw_angle(quaternion: Quaternion):
    # Convert the quaternion to a rotation matrix
    rotation_matrix = quaternion.rotation_matrix
    
    # Extract the yaw (azimuth) angle from the rotation matrix
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return yaw

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew[:,3:]

def build_two_point_clouds(genrtd_pcd, object_pcd):
    pcd_pred = o3d.geometry.PointCloud()
    c_pred = genrtd_pcd.cpu().detach().numpy()
    pcd_pred.points = o3d.utility.Vector3dVector(c_pred)

    pcd_gt = o3d.geometry.PointCloud()
    g_pred = object_pcd.cpu().detach().numpy()
    pcd_gt.points = o3d.utility.Vector3dVector(g_pred)
    return pcd_pred, pcd_gt