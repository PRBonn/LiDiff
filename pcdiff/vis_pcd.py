import open3d as o3d
import numpy as np
import click
        
                                                                                                                                     
@click.command()
@click.option('--path', '-p', type=str, help='path to pcd')
@click.option('--radius', '-r', type=float, default=50., help='range to filter pcd')
def main(path, radius):
    pcd = o3d.io.read_point_cloud(path)

    if radius > 0.:
        points = np.array(pcd.points)
        dist = np.sum(points**2, -1)**0.5
        pcd.points = o3d.utility.Vector3dVector(points[(dist < radius) & (points[:,-1] < 3.) & (points[:,-1] > -2.5)])

    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])
                                                                                                                                     
if __name__ == '__main__':
    main()
