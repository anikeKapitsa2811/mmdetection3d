"""Clean, runnable visualization examples with headless-safe output.

This file mirrors key snippets from vis.py and makes behavior explicit for
GUI vs headless environments.
"""

import os
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmengine import load

from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer

HAS_DISPLAY = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
OUT_DIR = Path('demo/vis_outputs')


def show_image_or_save(visualizer: Det3DLocalVisualizer, out_file: Path) -> None:
    """Show image when GUI is available, otherwise save image to disk."""
    if HAS_DISPLAY:
        visualizer.show()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_rgb = visualizer.get_image()
    mmcv.imwrite(mmcv.imconvert(img_rgb, 'rgb', 'bgr'), str(out_file))
    print(f'No GUI display detected. Saved visualization to: {out_file}')


def show_o3d_hint(visualizer: Det3DLocalVisualizer) -> None:
    """For Open3D point cloud windows, print a clear headless message."""
    if HAS_DISPLAY:
        visualizer.show()
    else:
        print('No GUI display detected. Open3D window from visualizer.show() is not visible here.')


def draw_points_on_image_example() -> None:
    info_file = load('demo/data/kitti/000008.pkl')
    points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32).reshape(-1, 4)[:, :3]
    lidar2img = np.array(info_file['data_list'][0]['images']['CAM2']['lidar2img'], dtype=np.float32)

    visualizer = Det3DLocalVisualizer()
    img = mmcv.imread('demo/data/kitti/000008.png')
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.set_image(img)
    visualizer.draw_points_on_image(points, lidar2img)
    show_image_or_save(visualizer, OUT_DIR / 'points_on_image.png')


def draw_3d_boxes_on_point_cloud_example() -> None:
    points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32).reshape(-1, 4)

    visualizer = Det3DLocalVisualizer()
    visualizer.set_points(points)

    # LiDARInstance3DBoxes expects shape [N, 7], even for one box.
    bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900, -1.5808]], dtype=torch.float32)
    )
    visualizer.draw_bboxes_3d(
        bboxes_3d, bbox_color=np.array([[0, 255, 0]], dtype=np.float64))
    show_o3d_hint(visualizer)


def draw_projected_3d_boxes_on_image_example() -> None:
    info_file = load('demo/data/kitti/000008.pkl')
    cam2img = np.array(info_file['data_list'][0]['images']['CAM2']['cam2img'], dtype=np.float32)

    bboxes_3d = [instance['bbox_3d'] for instance in info_file['data_list'][0]['instances']]
    gt_bboxes_3d = CameraInstance3DBoxes(np.array(bboxes_3d, dtype=np.float32))

    visualizer = Det3DLocalVisualizer()
    img = mmcv.imread('demo/data/kitti/000008.png')
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.set_image(img)

    visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, {'cam2img': cam2img})
    show_image_or_save(visualizer, OUT_DIR / 'proj_boxes_on_image.png')


if __name__ == '__main__':
    draw_points_on_image_example()
    draw_3d_boxes_on_point_cloud_example()
    draw_projected_3d_boxes_on_image_example()
