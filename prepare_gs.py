"""Convert .ply files into the ShapeNet-style format.

This script creates point clouds with 1,024 and 2,048 points and renders 24
random views for each model. The output directory structure matches the
ShapeNet dataset so that the training code can consume the data directly.
"""

import os
import argparse
import random
from typing import Iterable

import numpy as np
import open3d as o3d


def normalize_points(points):
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    return points


def sample_points(points, num):
    if len(points) >= num:
        idx = np.random.choice(len(points), num, replace=False)
    else:
        idx = np.random.choice(len(points), num, replace=True)
    return points[idx]


def iter_ply_files(path: str) -> Iterable[tuple[str, str]]:
    """Yield tuples of (category, ply_path)."""
    if os.path.isfile(path) and path.lower().endswith(".ply"):
        category = os.path.basename(os.path.dirname(path))
        yield category, path
        return

    if any(fname.lower().endswith(".ply") for fname in os.listdir(path)):
        category = os.path.basename(path)
        for fname in os.listdir(path):
            if fname.lower().endswith(".ply"):
                yield category, os.path.join(path, fname)
        return

    for category in sorted(os.listdir(path)):
        cat_path = os.path.join(path, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            if fname.lower().endswith(".ply"):
                yield category, os.path.join(cat_path, fname)


def render_views(pcd, out_dir, num_views=24, width=224, height=224):
    """Render ``num_views`` random viewpoints of ``pcd`` to ``out_dir``."""
    import open3d.visualization.rendering as rendering

    os.makedirs(out_dir, exist_ok=True)
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    center = pcd.get_center()
    radius = np.max(np.linalg.norm(np.asarray(pcd.points) - center, axis=1))

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.add_geometry("pcd", pcd, material)

    for i in range(num_views):
        theta = np.arccos(2 * random.random() - 1)
        phi = 2 * np.pi * random.random()
        eye = center + radius * 2.5 * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        renderer.scene.camera.look_at(center, eye, [0, 0, 1])
        img = renderer.render_to_image()
        o3d.io.write_image(os.path.join(out_dir, f"{i:02}.png"), img)

    renderer.release()


def process_ply(ply_path, pc_dir, render_dir):
    """Create point clouds and renderings for a single ``.ply`` file."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    points = normalize_points(points)

    np.save(os.path.join(pc_dir, "pointcloud_1024.npy"), sample_points(points, 1024))
    np.save(os.path.join(pc_dir, "pointcloud_2048.npy"), sample_points(points, 2048))

    pcd.points = o3d.utility.Vector3dVector(points)
    render_views(pcd, render_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare gs data in ShapeNet format")
    parser.add_argument(
        "input_path", help="Path to a gs directory or a single .ply file"
    )
    parser.add_argument(
        "output_dir", default="data", nargs="?", help="Output root directory"
    )
    args = parser.parse_args()

    for category, ply_path in iter_ply_files(args.input_path):
        model_name = os.path.splitext(os.path.basename(ply_path))[0]
        pc_dir = os.path.join(args.output_dir, "ShapeNet_pointclouds", category, model_name)
        render_dir = os.path.join(
            args.output_dir,
            "ShapeNetRendering",
            category,
            model_name,
            "rendering",
        )
        os.makedirs(pc_dir, exist_ok=True)
        os.makedirs(render_dir, exist_ok=True)
        process_ply(ply_path, pc_dir, render_dir)


if __name__ == "__main__":
    main()

