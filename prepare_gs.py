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
import torch
from PIL import Image
from gsplat import rasterization


def load_scaniverse_ply(path: str) -> np.ndarray:
    """Load Scaniverse-style PLY file and return the raw vertex data."""
    with open(path, "rb") as f:
        num_vertices = None
        while True:
            line = f.readline().decode("ascii")
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[2])
            if line.startswith("end_header"):
                break
        if num_vertices is None:
            raise ValueError("Invalid PLY: vertex count missing")
        data = np.fromfile(f, dtype="<f4").reshape(num_vertices, -1)
    return data


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


def render_views(data, out_dir, num_views=24, width=224, height=224):
    """Render ``num_views`` random viewpoints of Gaussian data to ``out_dir``."""

    os.makedirs(out_dir, exist_ok=True)

    pos = torch.from_numpy(data[:, :3]).float().cuda()
    sh = torch.from_numpy(data[:, 6 : 6 + 3 + 43]).float().cuda()
    opacity = torch.from_numpy(data[:, 6 + 43 + 1]).float().cuda()
    scale = torch.from_numpy(data[:, 6 + 43 + 2 : 6 + 43 + 5]).float().cuda()
    rot = torch.from_numpy(data[:, 6 + 43 + 5 : 6 + 43 + 9]).float().cuda()

    center = pos.mean(dim=0).cpu().numpy()
    radius = torch.norm(pos - pos.mean(dim=0), dim=1).max().item()

    device = pos.device

    def look_at(eye: np.ndarray, target: np.ndarray) -> torch.Tensor:
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        rot = np.stack([right, up, forward], axis=1)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = eye
        return torch.from_numpy(c2w).to(pos.device)

    # Intrinsic matrix used for all renders
    intrinsics = (
        torch.tensor(
            [[800.0, 0.0, width / 2], [0.0, 800.0, height / 2], [0.0, 0.0, 1.0]],
            device=device,
        )
        .unsqueeze(0)
    )

    for i in range(num_views):
        theta = np.arccos(2 * random.random() - 1)
        phi = 2 * np.pi * random.random()
        eye = center + radius * 2.5 * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        c2w = look_at(eye.astype(np.float32), center.astype(np.float32)).unsqueeze(0)
        # N = sh.shape[0]
        # K = (3 + 1)**2  # 16
        # sh = sh.view(N, K, 3)
        rgb, _, _ = rasterization(
            pos,
            rot,
            scale,
            opacity,
            sh,
            c2w,
            intrinsics,
            width,
            height,
            render_mode="RGB"   
        )
        img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:02}.png"))


def process_ply(ply_path, pc_dir, render_dir):
    """Create point clouds and renderings for a single ``.ply`` file."""
    data = load_scaniverse_ply(ply_path)
    points = data[:, :3]
    normalized = normalize_points(points)

    np.save(os.path.join(pc_dir, "pointcloud_1024.npy"), sample_points(normalized, 1024))
    np.save(os.path.join(pc_dir, "pointcloud_2048.npy"), sample_points(normalized, 2048))

    render_views(data, render_dir)


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

