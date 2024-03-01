import numpy as np
import open3d as o3d
import hdbscan
import matplotlib.pyplot as plt
import MinkowskiEngine as ME
import os

def overlap_clusters(cluster_i, cluster_j, min_cluster_point=10):
    # get unique labels from pcd_i and pcd_j from segments bigger than min_clsuter_point
    unique_i, count_i = np.unique(cluster_i, return_counts=True)
    unique_i = unique_i[count_i > min_cluster_point]

    unique_j, count_j = np.unique(cluster_j, return_counts=True)
    unique_j = unique_j[count_j > min_cluster_point]

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

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

def apply_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3]

def undo_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:,:3]

def aggregate_pcds(data_batch, data_dir, t_frame):
    # load empty pcd point cloud to aggregate
    pcd_full = np.empty((0,3))
    pcd_part = None

    # define "namespace"
    seq_num = data_batch[0].split('/')[-3]
    fname = data_batch[0].split('/')[-1].split('.')[0]

    # load poses
    datapath = data_batch[0].split('velodyne')[0]
    poses = load_poses(os.path.join(datapath, 'calib.txt'), os.path.join(datapath, 'poses.txt'))

    for t in range(len(data_batch)):
        # load the next t scan and aggregate
        fname = data_batch[t].split('/')[-1].split('.')[0]

        # load the next t scan, apply pose and aggregate
        p_set = np.fromfile(data_batch[t], dtype=np.float32)
        p_set = p_set.reshape((-1, 4))[:,:3]

        label_file = data_batch[t].replace('velodyne', 'labels').replace('.bin', '.label')
        l_set = np.fromfile(label_file, dtype=np.uint32)
        l_set = l_set.reshape((-1))
        l_set = l_set & 0xFFFF

        # remove moving points
        static_idx = l_set < 252
        p_set = p_set[static_idx]

        # remove flying artifacts
        dist = np.power(p_set, 2)
        dist = np.sqrt(dist.sum(-1))
        p_set = p_set[dist > 3.5]

        pose_idx = int(fname)
        p_set = apply_transform(p_set, poses[pose_idx])

        if t == t_frame:
            # will be aggregated later to the full pcd
            pcd_part = p_set.copy()
        else:
            pcd_full = np.vstack([pcd_full, p_set])


    # get start position of each aggregated pcd

    pose_idx = int(fname)
    pcd_full = undo_transform(pcd_full, poses[pose_idx])
    pcd_part = undo_transform(pcd_part, poses[pose_idx])

    return pcd_full, pcd_part

def clusters_hdbscan(points_set, n_clusters=50):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, ground):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # instead of ransac use patchwork
    inliers = list(np.where(ground == 9)[0])

    pcd_ = pcd.select_by_index(inliers, invert=True)
    labels_ = np.expand_dims(clusters_hdbscan(np.asarray(pcd_.points)), axis=-1)

    # that is a blessing of array handling
    # pcd are an ordered list of points
    # in a list [a, b, c, d, e] if we get the ordered indices [1, 3]
    # we will get [b, d], however if we get ~[1, 3] we will get the opposite indices
    # still ordered, i.e., [a, c, e] which means listing the inliers indices and getting
    # the invert we will get the outliers ordered indices (a sort of indirect indices mapping)
    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    return labels

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]

def visualize_pcd_clusters(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:,-1])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[labels[:,-1] == flat_indices[int(idx)]] = colors_instance[int(idx)]

    colors[labels[:,-1] == -1] = [0.,0.,0.,0.]

    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

    o3d.visualization.draw_geometries([pcd])
