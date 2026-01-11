"""
    vis a single image
"""
import os
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision

import albumentations as A

from model import UNet
from utils.lmds import LMDS
from model.utils import load_model


def args_parser():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--bilinear', default=True, type=bool)

    # device
    parser.add_argument('--device', default='cuda', type=str)

    # path
    parser.add_argument("--img_path", default="red_square.png", type=str)
    parser.add_argument("--output_path", default="vis", type=str)
    parser.add_argument("--checkpoint_path", default="weights/no_sigmoid/best_model.pth", type=str)

    # lmds
    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.1, type=float)

    return parser.parse_args()


def vis(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    image = Image.open(args.img_path).convert('RGB') # PIL numpy.ndarray (H, W, 3)
    transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
    ])
    image_np_ = np.array(image)
    image_np = transform(image=image_np_)['image'] # (H, W, 3)
    image = torchvision.transforms.ToTensor()(image_np) # tensor (3, H, W)
    image = image.unsqueeze(0).to(device) # tensor (1, 3, H, W)

    # model
    model = UNet(num_ch=3, num_class=args.num_classes, bilinear=args.bilinear)
    model.to(device)
    model = load_model(model, args.checkpoint_path, strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(image) # [B, 2, H, W]
        # outputs = outputs[:, 1:2, :, :]
        # outputs = torch.sigmoid(outputs)

    lmds = LMDS(
        kernel_size=(args.lmds_kernel_size, args.lmds_kernel_size),
        adapt_ts=args.lmds_adapt_ts
    )

    counts, locs, labels, scores = lmds(outputs) # counts: [[fg, bg]]

    locs_pred = np.asarray(locs[0])

    draw_img = cv2.cvtColor(image_np_, cv2.COLOR_RGB2BGR)

    for p, label in zip(locs[0], labels[0]):
        if label == 1:
            y, x = int(p[0]), int(p[1])
            cv2.circle(draw_img, (x, y), 2, (0, 0, 255), -1)

    save_path = os.path.join(
        args.output_path,
        os.path.basename(args.img_path)
    )
    cv2.imwrite(save_path, draw_img)
    print(f"[INFO] count: {counts[0][0]}")
    print(f"[INFO] saved to: {save_path}")


if __name__ == '__main__':
    args = args_parser()

    vis(args)
