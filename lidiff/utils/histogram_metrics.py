import numpy as np
import open3d as o3d
from scipy.spatial.distance import jensenshannon
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
import matplotlib.pyplot as plt

def histogram_point_cloud(pcd, resolution, max_range, bev=False):
    # get bins size by the number of voxels in the pcd
    bins = int(2 * max_range / resolution)

    hist = np.histogramdd(pcd, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range]))

    return np.clip(hist[0], a_min=0., a_max=1.) if bev else hist[0]

def compute_jsd(hist_gt, hist_pred, bev=False, visualize=False):
    bev_gt = hist_gt.sum(-1) if bev else hist_gt
    norm_bev_gt = bev_gt / bev_gt.sum()
    norm_bev_gt = norm_bev_gt.flatten()

    bev_pred = hist_pred.sum(-1) if bev else hist_pred
    norm_bev_pred = bev_pred / bev_pred.sum()
    norm_bev_pred = norm_bev_pred.flatten()
    
    if visualize:
        # for visualization purposes
        grid = np.meshgrid(np.arange(len(hist_gt)), np.arange(len(hist_gt)))
        points = np.concatenate((grid[0].flatten()[:,None], grid[1].flatten()[:,None]), axis=-1)
        points = np.concatenate((points, np.zeros((len(points),1))),axis=-1)

        # build bev histogram gt view
        norm_hist_gt = bev_gt / bev_gt.max()
        colors_gt = plt.get_cmap('viridis')(norm_hist_gt)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt.reshape(-1,4)[:,:3])

        # build bev histogram pred view
        norm_hist_pred = bev_pred / bev_pred.max()
        colors_pred = plt.get_cmap('viridis')(norm_hist_pred)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred.reshape(-1,4)[:,:3])

    return jensenshannon(norm_bev_gt, norm_bev_pred)


def compute_hist_metrics(pcd_gt, pcd_pred, bev=False):
    hist_pred = histogram_point_cloud(np.array(pcd_pred.points), 0.5, 50., bev)
    hist_gt = histogram_point_cloud(np.array(pcd_gt.points), 0.5, 50., bev)
    
    return compute_jsd(hist_gt, hist_pred, bev)

def compute_chamfer(pcd_pred, pcd_gt):
    chamfer_distance = ChamferDistance()
    chamfer_distance.update(pcd_gt, pcd_pred)
    cd_pred_mean, cd_pred_std = chamfer_distance.compute()

    return cd_pred_mean

def compute_precision_recall(pcd_pred, pcd_gt):
    precision_recall = PrecisionRecall(0.05,2*0.05,100)
    precision_recall.update(pcd_gt, pcd_pred)
    pr, re, f1 = precision_recall.compute_auc()

    return pr, re, f1 

def preprocess_pcd(pcd):
    points = np.array(pcd.points)
    dist = np.sqrt(np.sum(points**2, axis=-1))
    pcd.points = o3d.utility.Vector3dVector(points[dist < 30.])

    return pcd

def compute_metrics(pred_path, gt_path):
    pcd_pred = preprocess_pcd(o3d.io.read_point_cloud(pred_path))
    points_pred = np.array(pcd_pred.points)
    pcd_gt = preprocess_pcd(o3d.io.read_point_cloud(gt_path))
    points_gt = np.array(pcd_gt.points)

    jsd_pred = compute_hist_metrics(points_pred, points_gt)

    cd_pred = compute_chamfer(pcd_pred, pcd_gt)

    pr_pred, re_pred, f1_pred = compute_precision_recall(pcd_pred, pcd_gt)

    return cd_pred, pr_pred, re_pred, f1_pred

