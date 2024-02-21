import open3d as o3d
import numpy as np
import scipy

MESHTYPE = 6
TETRATYPE = 10
PCDTYPE = 1

class Metrics3D():
    def prediction_is_empty(self, geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                empty_v = self.is_empty(len(geom.vertices))
                empty_t = self.is_empty(len(geom.triangles))
                empty = empty_t or empty_v
            elif geom_type == PCDTYPE:
                empty = self.is_empty(len(geom.points))
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            empty = self.is_empty(len(geom[:, :3]))
        elif isinstance(geom, torch.Tensor):
            empty = self.is_empty(len(geom[:, :3]))
        else:
            assert False, '{} type not supported'.format(type(geom))

        return empty

    @staticmethod
    def convert_to_pcd(geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                geom_pcd = geom.sample_points_uniformly(1000000)
            elif geom_type == PCDTYPE:
                geom_pcd = geom
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        elif isinstance(geom, torch.Tensor):
            geom = geom.detach().cpu().numpy()
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        else:
            assert False, '{} type not supported'.format(type(geom))

        return geom_pcd

    @staticmethod
    def is_empty(length):
        empty = True
        if length:
            empty = False
        return empty

        input()

class RMSE():
    def __init__(self):
        self.dists = []

        return

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        self.dists.append(np.mean(dist_pt_2_gt))

    def reset(self):
        self.dists = []

    def compute(self):
        dist = np.array(self.dists)
        return dist.mean(), dist.std()

class CompletionIoU():
    def __init__(self, voxel_sizes=[0.5, 0.2, 0.1]):
        self.voxel_sizes = voxel_sizes
        # num_thresholds, tp, fp, fn
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint64)

    def update(self, gt, pred):
        max_range = 50.
        for i, vsize in enumerate(self.voxel_sizes):
            bins = int(2 * max_range / vsize)
            vox_coords_gt = np.round(np.array(gt.points) / vsize).astype(int)
            hist_gt = np.histogramdd(
                    vox_coords_gt, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range])
            )[0].astype(bool).astype(int)
        
            vox_coords_pred = np.round(np.array(pred.points) / vsize).astype(int)
            hist_pred = np.histogramdd(
                    vox_coords_pred, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range])
            )[0].astype(bool).astype(int)

            self.conf_matrix[i][0] += ((hist_gt == 1) & (hist_pred == 1)).sum() # tp
            self.conf_matrix[i][1] += ((hist_gt == 1) & (hist_pred == 0)).sum() # fn
            self.conf_matrix[i][2] += ((hist_gt == 0) & (hist_pred == 1)).sum() # fp

    def compute(self):
        res_vsizes = {}
        for i, vsize in enumerate(self.voxel_sizes):
            tp = self.conf_matrix[i][0]
            fn = self.conf_matrix[i][1]
            fp = self.conf_matrix[i][2]
            
            intersection = tp
            union = tp + fn + fp + 1e-15

            res_vsizes[vsize] = intersection / union

        return res_vsizes

    def reset(self):
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint)

class ChamferDistance():
    def __init__(self):
        self.dists = []

        return

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        self.dists.append((np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2)

    def reset(self):
        self.dists = []

    def compute(self):
        cdist = np.array(self.dists)
        return cdist.mean(), cdist.std()

class PrecisionRecall(Metrics3D):

    def __init__(self, min_t, max_t, num):
        self.thresholds = np.linspace(min_t, max_t, num)
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def update(self, gt_pcd, pt_pcd):
        # precision: predicted --> ground truth
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        # recall: ground truth --> predicted
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        for t in self.thresholds:
            p = np.where(dist_pt_2_gt < t)[0]
            p = 100 / len(dist_pt_2_gt) * len(p)
            self.pr_dict[t].append(p)

            r = np.where(dist_gt_2_pt < t)[0]
            r = 100 / len(dist_gt_2_pt) * len(r)
            self.re_dict[t].append(r)

            # fscore
            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p + r)
            self.f1_dict[t].append(f)

    def reset(self):
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def compute_at_threshold(self, threshold):
        t = self.find_nearest_threshold(threshold)
        # print('computing metrics at threshold:', t)
        pr = sum(self.pr_dict[t]) / len(self.pr_dict[t])
        re = sum(self.re_dict[t]) / len(self.re_dict[t])
        f1 = sum(self.f1_dict[t]) / len(self.f1_dict[t])
        # print('precision: {}'.format(pr))
        # print('recall: {}'.format(re))
        # print('fscore: {}'.format(f1))
        return pr, re, f1, t

    def compute_auc(self):
        dx = self.thresholds[1] - self.thresholds[0]
        perfect_predictor = scipy.integrate.simpson(np.ones_like(self.thresholds), dx=dx)

        pr, re, f1 = self.compute_at_all_thresholds()

        pr_area = scipy.integrate.simpson(pr, dx=dx)
        norm_pr_area = pr_area / perfect_predictor

        re_area = scipy.integrate.simpson(re, dx=dx)
        norm_re_area = re_area / perfect_predictor

        f1_area = scipy.integrate.simpson(f1, dx=dx)
        norm_f1_area = f1_area / perfect_predictor

        # print('computing area under curve')
        # print('precision: {}'.format(norm_pr_area))
        # print('recall: {}'.format(norm_re_area))
        # print('fscore: {}'.format(norm_f1_area))

        return norm_pr_area, norm_re_area, norm_f1_area

    def compute_at_all_thresholds(self):
        pr = [sum(self.pr_dict[t]) / len(self.pr_dict[t]) for t in self.thresholds]
        re = [sum(self.re_dict[t]) / len(self.re_dict[t]) for t in self.thresholds]
        f1 = [sum(self.f1_dict[t]) / len(self.f1_dict[t]) for t in self.thresholds]
        return pr, re, f1

    def find_nearest_threshold(self, value):
        idx = (np.abs(self.thresholds - value)).argmin()
        return self.thresholds[idx]

