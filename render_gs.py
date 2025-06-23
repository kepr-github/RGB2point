"""Render Gaussian Splat data from a PLY file using gsplat.

This example shows how to load Scaniverse-style PLY files and render them
with ``gsplat.rasterization``.

Example:
    python render_gs.py your_scaniverse.ply output.png
"""

import argparse
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image


def load_scaniverse_ply(path: str) -> np.ndarray:
    """Load Scaniverse PLY file and return the raw vertex data."""
    with open(path, "rb") as f:
        num_vertices = None
        # parse header
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


def main():
    parser = argparse.ArgumentParser(description="Render a Gaussian PLY file")
    parser.add_argument("ply_path", help="Input Scaniverse PLY file")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    data = load_scaniverse_ply(args.ply_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos = torch.from_numpy(data[:, :3]).float().to(device)
    sh = torch.from_numpy(data[:, 6 : 6 + 3 + 43]).float().to(device)
    opacity = torch.from_numpy(data[:, 6 + 43 + 1]).float().to(device)
    scale = torch.from_numpy(data[:, 6 + 43 + 2 : 6 + 43 + 5]).float().to(device)
    rot = torch.from_numpy(data[:, 6 + 43 + 5 : 6 + 43 + 9]).float().to(device)

    cam_pose = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[800.0, 0.0, args.width / 2], [0.0, 800.0, args.height / 2], [0.0, 0.0, 1.0]],
        device=device,
    ).unsqueeze(0)

    rgb, _, _ = rasterization(
        pos.unsqueeze(0),
        rot.unsqueeze(0),
        scale.unsqueeze(0),
        opacity.unsqueeze(0),
        sh.unsqueeze(0),
        cam_pose,
        intrinsics,
        args.width,
        args.height,
    )

    img = (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(img).save(args.output)


if __name__ == "__main__":
    main()
