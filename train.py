
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.optim as optim
from PIL import Image
import numpy as np
from glob import glob
from accelerate import Accelerator
import argparse
import os
from datetime import datetime, timezone, timedelta
import yaml

from chamferdist import ChamferDistance
from tqdm import tqdm

from utils import PCDataset, chamfer_distance, EMDLoss, fscore
from model import PointCloudNet



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGB2Point")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--root", default=None, help="Dataset root directory")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="List of category names to use",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = args.root if args.root else cfg.get("dataset", {}).get("root", "data")
    categories = args.categories if args.categories else cfg.get("dataset", {}).get(
        "categories", ["02958343", "02691156", "03001627"]
    )

    accelerator = Accelerator(log_with="wandb")
    # Avoid shared memory exhaustion when using DataLoader workers
    mp.set_sharing_strategy("file_system")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    batch_size = cfg.get("training", {}).get("batch_size", 4)
    device = accelerator.device

    model = PointCloudNet(
        num_views=cfg.get("model", {}).get("num_views", 1),
        point_cloud_size=cfg.get("model", {}).get("point_cloud_size", 1024),
        num_heads=cfg.get("model", {}).get("num_heads", 4),
        dim_feedforward=cfg.get("model", {}).get("dim_feedforward", 2048),
    )
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("training", {}).get("learning_rate", 5e-4)))
    sched_cfg = cfg.get("training", {}).get("scheduler", {})
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.7),
        patience=sched_cfg.get("patience", 5),
        min_lr=float(sched_cfg.get("min_lr", 1e-5)),
        verbose=True,
        threshold=sched_cfg.get("threshold", 0.01),
    )

    num_epochs = cfg.get("training", {}).get("num_epochs", 1000)

    accelerator.init_trackers(project_name="wacv_pc1024", config={})

    # create timestamped directory for checkpoints
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("ckpt", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    chamferDist = ChamferDistance()
    label_table = {
        "02691156": "airplane",
        "02828884": "bench",
        "04379243": "table",
        "02933112": "cabinet",
        "02958343": "car",
        "03001627": "chair",
        "03211117": "display",
        "03636649": "lamp",
        "03691459": "loudspeaker",
        "04090263": "rifle",
        "04256520": "sofa",
        "04379243": "table",
        "04401088": "telephone",
        "04530566": "watercraft",
        "soybean":"soybean",
        "plantonly":"soybean1",
        "plantonly2":"soybean2",
    }


    dataset = PCDataset(stage="train", transform=transform, root=root, categories=categories)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
    )
    test_dataset = PCDataset(stage="test", transform=transform, root=root, categories=categories)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model, optimizer, dataloader, test_dataloader, sche = accelerator.prepare(
        model, optimizer, dataloader, test_dataloader, sche
    )

    best = 10000
    mse = nn.MSELoss(reduction="mean")


    

    mae_loss = nn.L1Loss()
    emd_loss = EMDLoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loss_history = []
        unet_loss_history = []
        iou_loss_history = []
        uni_loss_history = []
        mse_history = []
        p_history = []
        cd_history = []
        radius = 0.01

        """
        Training
        """
        for idx, (images, gt_pc, name) in enumerate(dataloader):
            gt_pc = gt_pc.float().to(device)
            images = images.to(device)
            optimizer.zero_grad()
            batch_loss = 0.0
            out = model(images)
            cd_loss = chamferDist(out, gt_pc, bidirectional=True) * 5.0
            loss = cd_loss
            cd_history.append(cd_loss.item())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            iou_loss_history.append(loss.item())

            if idx % 50 == 0:
                accelerator.print(
                    f"[Train|{epoch+1}] {idx}/{len(dataloader)} loss:{loss.item():.4f}  cd_loss:{cd_loss.item():.4f} "
                )
                accelerator.log(
                    {
                        "train_batch/loss": loss.item(),
                        "train_batch/cd_loss": cd_loss.item(),
                    }
                )
        accelerator.print(
            f"[Train]Epoch {epoch + 1}, Loss:{np.mean(iou_loss_history):.4f} "
        )
        accelerator.log(
            {
                "train/loss": np.mean(iou_loss_history),
                "train/cd_loss": np.mean(cd_history),
                "train/epoch": epoch + 1,
            }
        )
        model.eval()

        total_loss = 0.0
        loss_history = []
        unet_loss_history = []
        iou_loss_history = []
        category_table = {}
        gt_point = []
        pred_point = []
        cd_values = []
        result = []
        fscore_table = {}
        cd_table = {}
        """
        Testing
        """
        print(test_dataloader)
        for idx, (images, gt_pc, names) in tqdm(enumerate(test_dataloader)):
            gt_pc = gt_pc.float().to(device)
            images = images.to(device)
            batch_loss = 0.0
            with torch.no_grad():
                out = model(images)

            cd_loss = chamferDist(out, gt_pc, bidirectional=True) * 5.0
            cd_values.append(cd_loss.item())

            loss = cd_loss
            loss_history.append(loss.item())
            distance = chamfer_distance(
                out[0].detach().cpu().numpy(), gt_pc[0].detach().cpu().numpy()
            )

            result.append(distance)
            category = names[0].split("_")[0]
            if category not in cd_table:
                cd_table[category] = []
            cd_table[category].append(distance)

        accelerator.print(f"[Test]Epoch {epoch + 1},  loss:{np.mean(loss_history):.4f}")

        f_mean_table = {}

        f_mean = []
        for key in fscore_table.keys():
            f_mean_table[key] = np.mean(fscore_table[key])
            f_mean.append(np.mean(f_mean_table[key]))
        cdtable = {}
        total_cd = 0
        for key in cd_table.keys():
            human_read_key = label_table[key]
            cdtable[human_read_key] = np.mean(cd_table[key])
            total_cd += cdtable[human_read_key]

        accelerator.log(
            {"test/loss": np.mean(loss_history), "cd": cdtable, "test/epoch": epoch + 1}
        )

        # score = np.mean(total_cd)
        score = np.mean(loss_history)
        model_save_name = os.path.join(
            save_dir, f"model_epoch{epoch + 1}_score{score:.4f}.pth"
        )
        sche.step(score)
        print(score)
        if score < best:
            best = score
            print("model updated")
            if isinstance(model, nn.DataParallel):
                data = {
                    "model": model.module.state_dict(),
                }
                torch.save(data, model_save_name)
            else:
                data = {
                    "model": model.state_dict(),
                }
                torch.save(data, model_save_name)
