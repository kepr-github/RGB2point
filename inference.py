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
parser.add_argument("--dir", default=None, help="Input directory containing images")
parser.add_argument("--output_dir", default=None, help="Directory to save results when using --dir")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

model_path = args.model if args.model else cfg.get("inference", {}).get("model_path", "ckpt/mymodel.pth")

input_dir = args.dir if args.dir else cfg.get("inference", {}).get("image_dir")

if input_dir:
    output_dir = args.output_dir or cfg.get("inference", {}).get("output_dir")
    if not output_dir:
        default_save = cfg.get("inference", {}).get("save_path")
        output_dir = os.path.dirname(default_save) if default_save else "result"
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
else:
    image_path = args.image if args.image else cfg.get("inference", {}).get("image_path", "img/09.png")
    if args.output:
        save_path = args.output
    else:
        default_save = cfg.get("inference", {}).get("save_path")
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

if input_dir:
    for img in image_files:
        image_path = os.path.join(input_dir, img)
        base_name = os.path.splitext(img)[0]
        save_path = os.path.join(output_dir, base_name + ".ply")
        predict(model, image_path, save_path)
else:
    predict(model, image_path, save_path)
