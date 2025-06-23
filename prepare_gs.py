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
    """Normalize point cloud to be within a unit sphere centered at the origin."""
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    return points


def sample_points(points, num):
    """Sample a specified number of points from the point cloud."""
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
    
    # Updated data slicing based on the provided PLY header structure.
    # Expected structure: 3(pos) + 3(norm) + 3(f_dc) + 45(f_rest) + 1(opacity) + 3(scale) + 4(rot) = 62 columns
    if data.shape[1] < 62:
        raise ValueError(f"Expected at least 62 columns based on PLY header, but got {data.shape[1]}")

    # --- 1. Data Preparation and Conversion ---
    
    # Normalize positions to be centered at the origin and fit within a unit sphere
    pos = torch.from_numpy(data[:, :3]).float().cuda()
    center = pos.mean(dim=0)
    pos = pos - center
    scale_factor = torch.max(torch.linalg.norm(pos, dim=1))
    pos = pos / scale_factor

    # Extract all 48 spherical harmonic coefficients (3 DC + 45 AC)
    # The data is reshaped from (N, 48) to (N, 16, 3) for gsplat (SH degree 3)
    sh = torch.from_numpy(data[:, 6:54]).float().cuda().reshape(-1, 16, 3)

    # Convert opacity from logit space using sigmoid
    opacity = torch.sigmoid(torch.from_numpy(data[:, 54])).float().cuda()

    # Convert scale from log space using exp and normalize by the same factor as position
    scale = torch.exp(torch.from_numpy(data[:, 55:58])).float().cuda() / scale_factor

    # Normalize rotation quaternions
    rot = torch.from_numpy(data[:, 58:62]).float().cuda()
    rot = rot / torch.linalg.norm(rot, dim=1, keepdim=True)
    
    device = pos.device
    
    # --- 2. Camera Setup ---

    def look_at(eye: np.ndarray, target: np.ndarray) -> torch.Tensor:
        """Computes a camera-to-world transformation matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Use Y-up convention, common in graphics
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6: # Handle case where forward is parallel to up
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([-right, up, forward], axis=1) # Corrected orientation
        c2w[:3, 3] = eye
        return torch.from_numpy(c2w).to(device)

    # Intrinsic matrix used for all renders
    focal_length = 1.8 # Adjust this based on desired zoom level
    intrinsics = torch.tensor(
        [[focal_length * width, 0.0, width / 2], 
         [0.0, focal_length * width, height / 2], 
         [0.0, 0.0, 1.0]],
        device=device,
    )

    # --- 3. Rendering Loop ---
    
    # After normalization, the object center is at (0,0,0) and radius is ~1.0
    view_center = np.array([0.0, 0.0, 0.0])
    view_radius = 1.0

    for i in range(num_views):
        # Sample camera position on a sphere
        # Sample theta only for the upper hemisphere so cameras never look from below
        theta = np.arccos(random.random())
        phi = 2 * np.pi * random.random()
        camera_dist = view_radius * 2.5 # Distance from the center
        eye = view_center + camera_dist * np.array(
            [np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)]
        )

        c2w = look_at(eye.astype(np.float32), view_center.astype(np.float32))
        # The 'viewmats' argument requires the world-to-camera matrix, which is the inverse of camera-to-world.
        viewmat = torch.inverse(c2w).unsqueeze(0)
        
        # Render the image
        # NOTE: Keyword arguments have been updated to match the new gsplat function signature.
        rgb, _, _ = rasterization(
            means=pos,
            quats=rot,
            scales=scale,
            opacities=opacity,
            colors=sh,
            viewmats=viewmat,
            Ks=intrinsics.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=3, # Explicitly set the spherical harmonics degree.
            render_mode="RGB",
        )

        # Convert to a displayable image format
        # FIX: Removed incorrect transpose operation. gsplat likely returns (H, W, C) which is what PIL expects.
        img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
        
        # --- 4. Debugging and Saving ---
        print(f"Rendered view {i:02}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:02}.png"))


def process_ply(ply_path, pc_dir, render_dir):
    """Create point clouds and renderings for a single ``.ply`` file."""
    print(f"Processing {ply_path}...")
    try:
        data = load_scaniverse_ply(ply_path)
        points = data[:, :3]
        points[:, 1] *= -1  # Flip Y axis for point cloud generation
        normalized_points = normalize_points(points)

        os.makedirs(pc_dir, exist_ok=True)
        np.save(
            os.path.join(pc_dir, "pointcloud_1024.npy"), sample_points(normalized_points, 1024)
        )
        np.save(
            os.path.join(pc_dir, "pointcloud_2048.npy"), sample_points(normalized_points, 2048)
        )

        render_views(data, render_dir)
        print(f"Successfully processed and saved outputs to {pc_dir} and {render_dir}")
    except Exception as e:
        print(f"Failed to process {ply_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Gaussian Splatting data in ShapeNet format")
    parser.add_argument(
        "input_path", help="Path to a directory with .ply files or a single .ply file"
    )
    parser.add_argument(
        "output_dir", default="data", nargs="?", help="Output root directory"
    )
    args = parser.parse_args()

    for category, ply_path in iter_ply_files(args.input_path):
        model_name = os.path.splitext(os.path.basename(ply_path))[0]
        pc_dir = os.path.join(
            args.output_dir, "ShapeNet_pointclouds", category, model_name
        )
        render_dir = os.path.join(
            args.output_dir,
            "ShapeNetRendering",
            category,
            model_name,
            "rendering",
        )
        # No need to create directories here, process_ply will do it
        process_ply(ply_path, pc_dir, render_dir)


if __name__ == "__main__":
    main()