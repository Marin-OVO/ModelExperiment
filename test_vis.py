"""
    batch vis images
"""
import os
import csv
import argparse

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import albumentations as A

from model import UNet
from datasets import CrowdDataset
from datasets.transforms import DownSample
from utils.logger import time_str
from utils.lmds import LMDS
from utils.metrics import PointsMetrics
from model.utils import load_model


def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_root', default='data/crowdsat', type=str)
    parser.add_argument('--checkpoint_path', default='weights/no_sigmoid/best_model.pth', type=str)
    parser.add_argument('--output_path', default='vis', type=str)

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--bilinear', default=True, type=bool)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--ds_down_ratio', default=1, type=int)
    parser.add_argument('--ds_crowd_type', default='point', type=str)

    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.1, type=float)

    return parser.parse_args()


def vis(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestr = time_str()
    work_dir = os.path.join(args.output_path, f'vis_{timestr}')
    os.makedirs(work_dir, exist_ok=True)

    vis_dir = os.path.join(work_dir, 'vis')
    tf_dir = os.path.join(work_dir, 'vis_tf')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(tf_dir, exist_ok=True)

    # dataset
    test_dataset = CrowdDataset(
        data_root=args.data_root,
        train=False,
        train_list='crowd_train.list',
        val_list='crowd_val.list',
        albu_transforms=[
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        ],
        end_transforms=[
            DownSample(down_ratio=args.ds_down_ratio,
                       crowd_type=args.ds_crowd_type)
        ]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # model
    model = UNet(num_ch=3, num_class=args.num_classes, bilinear=args.bilinear)
    model = load_model(model, args.checkpoint_path, strict=False)
    model.to(device)
    model.eval()

    # lmds
    ks = args.lmds_kernel_size
    lmds = LMDS(kernel_size=(ks, ks), adapt_ts=args.lmds_adapt_ts)

    metrics = PointsMetrics(radius=args.radius, num_classes=args.num_classes)

    def draw_points(img, points, drawer, cfg):
        for p in points:
            x, y = int(p[0]), int(p[1])
            drawer(img, (x, y), **cfg)

    pred_draw_cfg = (
        cv2.circle,
        dict(radius=4, color=(0, 0, 255), thickness=-1)  # 红色预测点
    )

    draw_cfg = [
        # ("gt", cv2.circle, dict(
        #     radius=6,
        #     color=(0, 255, 0),
        #     thickness=2
        # )),
        ("tp", cv2.circle, dict(
            radius=4,
            color=(255, 255, 0),
            thickness=-1
        )),
        ("fp", cv2.circle, dict(
            radius=4,
            color=(255, 0, 255),
            thickness=2
        )),
        ("fn", cv2.drawMarker, dict(
            color=(0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=2
        ))
    ]

    with torch.no_grad():
        for idx, (image, target) in enumerate(test_loader):

            image = image.to(device)
            img_path = target['img_path'][0]

            raw_img = cv2.imread(img_path)
            H0, W0 = raw_img.shape[:2]

            gt_points = target['points'][0].cpu().numpy()  # (x, y)
            gt_loc = [(float(x), float(y)) for x, y in gt_points]
            gt = dict(
                loc=gt_loc,
                labels=[1] * len(gt_loc)
            )

            outputs = model(image)
            # outputs = torch.sigmoid(outputs)

            counts, locs, labels, scores = lmds(outputs)

            down_ratio = args.ds_down_ratio
            pred_loc = [
                (float(x) * down_ratio, float(y) * down_ratio)
                for (y, x), label in zip(locs[0], labels[0])
                if label == 1  # only fg
            ]
            pred = dict(
                loc=pred_loc,
                labels=[1] * len(pred_loc)
            )

            img_pred = raw_img.copy()
            drawer, cfg = pred_draw_cfg
            draw_points(img_pred, pred['loc'], drawer, cfg)

            cv2.imwrite(
                os.path.join(vis_dir, os.path.basename(img_path)),
                img_pred
            )

            metrics.flush()
            metrics.feed(gt=gt, preds=pred)

            tp = metrics.current_tp if metrics.current_tp else []
            fp = metrics.current_fp if metrics.current_fp else []
            fn = metrics.current_fn if metrics.current_fn else []

            # tf vis
            img_tf = raw_img.copy()
            for name, drawer, cfg in draw_cfg:
                # pts = dict(gt=gt['loc'], tp=tp, fp=fp, fn=fn)[name]
                pts = dict(tp=tp, fp=fp, fn=fn)[name]
                draw_points(img_tf, pts, drawer, cfg)

            cv2.imwrite(
                os.path.join(tf_dir, os.path.basename(img_path)),
                img_tf
            )
            print(f"[{idx + 1:3d}/{len(test_dataset)}] saved {img_path}")

    # save to csv
    csv_path = os.path.join(work_dir, 'test_metrics.csv')

    metrics.aggregate()

    recall = metrics.recall()
    precision = metrics.precision()
    f1_score = metrics.fbeta_score()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', precision])
        writer.writerow(['Recall', recall])
        writer.writerow(['F1-score', f1_score])

    print(f"[INFO] test metrics saved to {csv_path}")


if __name__ == '__main__':
    args = args_parser()

    vis(args)