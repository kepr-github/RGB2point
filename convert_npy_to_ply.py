import argparse
import numpy as np

from utils import export_to_ply


def main():
    parser = argparse.ArgumentParser(description="Convert a point cloud saved as a .npy file to PLY format")
    parser.add_argument("npy_path", help="Path to the input .npy file")
    parser.add_argument("ply_path", help="Path to the output .ply file")
    args = parser.parse_args()

    points = np.load(args.npy_path)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Input .npy file must contain an array of shape (N, 3) or (N, >=3)")

    export_to_ply(points[:, :3], args.ply_path)


if __name__ == "__main__":
    main()
