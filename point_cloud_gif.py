import numpy as np
import glob
import pyvista
import contextlib
from PIL import Image
import sys
from tqdm.contrib import tenumerate
import open3d as o3d
import os


def setup_plotter(interactive=False, resolution=[1920, 1080]):
    if interactive:
        plotter = pyvista.Plotter()
    else:
        plotter = pyvista.Plotter(off_screen=True, window_size=resolution)
    plotter.enable_eye_dome_lighting()
    plotter.enable_anti_aliasing(aa_type='ssaa', multi_samples=False, all_renderers=True)
    # plotter.enable_shadows()
    plotter.camera.SetViewAngle(50.)
    return plotter

def add_points(plotter: pyvista.Plotter, points, point_size=10.):
    plotter.add_points(points[:, :3], point_size=point_size, render_points_as_spheres=True, color='red', specular=.4, roughness=0.4)

def set_camera_position(plotter, position, focal_point, roll=0):
    plotter.camera.position = position
    plotter.camera.focal_point = focal_point
    plotter.camera.roll = roll
    print(f"Set camera position: {position}, focal point: {focal_point}, roll: {roll}")

def print_camera_info(plotter):
    print(f"Camera Position: {plotter.camera.position}")
    print(f"Camera Focal Point: {plotter.camera.focal_point}")
    print(f"Camera View Angle: {plotter.camera.view_angle}")
    print(f"Camera Roll: {plotter.camera.roll}")

def rgba_to_rgb_with_white_background(image):
    # Create a new image with white background
    rgb_image = Image.new("RGB", image.size, "white")
    
    # Paste the RGBA image onto the white background
    rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
    
    return rgb_image


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide the name of the folder containing the segments"
    segments_dir = sys.argv[1]
    resolution = [600, 600]
    files = glob.glob(f"{segments_dir}/*.ply")
    files = list(sorted(files, key=os.path.getmtime))
    
    print(f"Number of files: {len(files)}")
    plotter = setup_plotter(interactive=False, resolution=resolution)
    
    # Manually set different camera positions and angles
    camera_positions = [
        ([-5., 5., 0.], [0., 0., 0.], 90),
    ]
    
    for cam_pos, focal_pt, roll in camera_positions:
        print(f"Trying camera position: {cam_pos} with focal point: {focal_pt} and roll: {roll}")
        set_camera_position(plotter, cam_pos, focal_pt, roll)

        # Add a reference cube to visualize the camera's position
        # cube = pyvista.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
        # plotter.add_mesh(cube, color='blue', opacity=0.5)

        for i, file in tenumerate(files):
            o3_pc: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file)
            points = np.asarray(o3_pc.points)
            if points.size == 0:
                print(f"File {file} is empty or not read correctly.")
                continue

            colors = np.ones((len(points), 3)) * .5
            colors[:,] =[.3,1.,.3]
            o3_pc.colors = o3d.utility.Vector3dVector(colors)
            pc = np.concatenate((points, colors), axis=1)
            
            add_points(plotter, pc, point_size=7.)

            # Debug: Check if points are added to plotter
            if plotter.renderer.GetActors().GetNumberOfItems() == 0:
                print(f"No actors added for file {file}")
            else:
                plotter.screenshot(f"lidiff/random_pcds/generated_pcd/gif_visualization/test_{i:04d}_pos{cam_pos[1]}_roll{roll}.png", transparent_background=True, window_size=resolution)
            
            plotter.clear_actors()
        plotter.clear()

    plotter.deep_clean()
    
    fp_in = "lidiff/random_pcds/generated_pcd/gif_visualization/*.png"
    fp_out = "image.gif"
    fp_out_slower = "image_slower.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
        list_image = sorted(glob.glob(fp_in))
        # lazily load images
        imgs = (stack.enter_context(rgba_to_rgb_with_white_background(Image.open(f)))
                for f in list_image)

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=20 * len(list_image), loop=0)
