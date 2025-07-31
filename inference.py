from model import PointCloudNet
from utils import predict
import torch
import argparse
import yaml
import os



parser = argparse.ArgumentParser(description="RGB2Point Inference")
parser.add_argument("--config", default="config.yaml", help="Path to config file")
parser.add_argument("--model", default=None, help="Model checkpoint path")
parser.add_argument("--image", default=None, help="Input image")
parser.add_argument("--output", default=None, help="Output ply path")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

model_path = args.model if args.model else cfg.get("inference", {}).get("model_path", "ckpt/mymodel.pth")
image_path = args.image if args.image else cfg.get("inference", {}).get("image_path", "img/09.png")

if args.output:
    save_path = args.output
else:
    default_save = cfg.get("inference", {}).get("save_path")
    # Determine the directory where the ply file will be written
    if default_save:
        output_dir = os.path.dirname(default_save) or "."
    else:
        output_dir = "result"
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, base_name + ".ply")

model = PointCloudNet(
    num_views=cfg.get("model", {}).get("num_views", 1),
    point_cloud_size=cfg.get("model", {}).get("point_cloud_size", 1024),
    num_heads=cfg.get("model", {}).get("num_heads", 4),
    dim_feedforward=cfg.get("model", {}).get("dim_feedforward", 2048),
    train_vit=cfg.get("model", {}).get("train_vit", False),
)
model.load_state_dict(torch.load(model_path)["model"])
model.eval()

predict(model, image_path, save_path)

