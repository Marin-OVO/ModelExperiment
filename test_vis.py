import os
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

import albumentations as A

from model import UNet
from datasets import CrowdDataset
from datasets.transforms import DownSample
from utils.lmds import LMDS
from model.utils import load_model


def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_root', default='data/crowdsat', type=str)

    parser.add_argument('--checkpoint_path', default="weights/best_model.pth", type=str)
    parser.add_argument('--output_path', default='vis', type=str)

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--bilinear', default=True, type=bool)

    parser.add_argument('--device', default='cuda', type=str)

    # dataset / post-process
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--ds_down_ratio', default=1, type=int)
    parser.add_argument('--ds_crowd_type', default='point', type=str)

    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.5, type=float)

    args = parser.parse_args()

    return args


def match_points(pred_points, gt_points, radius):

    tp, fp, fn = [], [], []
    gt_used = np.zeros(len(gt_points), dtype=bool)

    for p in pred_points:
        dists = np.linalg.norm(gt_points - p, axis=1)
        min_idx = np.argmin(dists) if len(dists) > 0 else -1

        if min_idx >= 0 and dists[min_idx] <= radius and not gt_used[min_idx]:
            tp.append(p)
            gt_used[min_idx] = True
        else:
            fp.append(p)

    for i, used in enumerate(gt_used):
        if not used:
            fn.append(gt_points[i])

    return np.array(tp), np.array(fp), np.array(fn)


def vis(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    test_albu_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ]
    test_end_transforms = [
        DownSample(down_ratio=args.ds_down_ratio,
                   crowd_type=args.ds_crowd_type)
    ]

    test_dataset = CrowdDataset(
        data_root=args.data_root,
        train=False,
        train_list="crowd_train.list",
        val_list="crowd_val.list",
        albu_transforms=test_albu_transforms,
        end_transforms=test_end_transforms
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model = UNet(num_ch=3, num_class=args.num_classes, bilinear=args.bilinear)
    model.to(device)

    model = load_model(model, args.checkpoint_path, strict=False)
    model.eval()

    ks = args.lmds_kernel_size
    if isinstance(ks, int):
        ks = (ks, ks)

    lmds = LMDS(
        kernel_size=ks,
        adapt_ts=args.lmds_adapt_ts
    )

    def draw_points(img, points, drawer, cfg):
        for p in points:
            x, y = int(p[1]), int(p[0])
            drawer(img, (x, y), **cfg)

    draw_configs = [
        ("tp", cv2.circle, {
            "radius": 4,
            "color": (255, 255, 0),
            "thickness": -1
        }),
        ("fp", cv2.circle, {
            "radius": 4,
            "color": (255, 0, 255),
            "thickness": 2
        }),
        ("fn", cv2.drawMarker, {
            "color": (0, 255, 255),
            "markerType": cv2.MARKER_CROSS,
            "markerSize": 8,
            "thickness": 2
        })
    ]

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):

            image, target = batch

            img = image.to(device)

            gt_points = target["points"][0].cpu().numpy()
            img_path = target["img_path"][0]

            pred = model(img)
            _, locs, _, _ = lmds(pred)
            pred_points = np.asarray(locs[0], dtype=np.float32)

            tp, fp, fn = match_points(
                pred_points, gt_points, radius=args.radius
            )

            raw_img = cv2.imread(img_path)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            for name, drawer, cfg in draw_configs:
                pts = {"tp": tp, "fp": fp, "fn": fn}[name]
                draw_points(raw_img, pts, drawer, cfg)

            save_path = os.path.join(
                args.output_path,
                os.path.basename(img_path)
            )
            cv2.imwrite(save_path, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))

            print(f"[{idx + 1}/{len(test_dataset)}] saved {save_path}")


if __name__ == "__main__":
    args = args_parser()

    vis(args)
