import os
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from model import UNet
from utils.lmds import LMDS
from model.utils import load_model


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--bilinear', default=True, type=bool)

    # device
    parser.add_argument('--device', default='cuda', type=str)

    # path
    parser.add_argument("--img_path", default="red_square.png", type=str)
    parser.add_argument("--output_path", default="vis", type=str)
    parser.add_argument("--checkpoint_path", default="weights/best_model.pth", type=str)

    # LMDS
    parser.add_argument('--lmds_kernel_size', default=3, type=int)
    parser.add_argument('--lmds_adapt_ts', default=0.5, type=float)

    return parser.parse_args()


def vis(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    img_pil = Image.open(args.img_path).convert('RGB')
    img_np = np.array(img_pil)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    model = UNet(num_ch=3, num_class=args.num_classes, bilinear=args.bilinear).to(device)

    model = load_model(model, args.checkpoint_path, strict=False)
    model.eval()

    with torch.no_grad():
        pred = model(img_tensor)

    lmds = LMDS(
        kernel_size=(args.lmds_kernel_size, args.lmds_kernel_size),
        adapt_ts=args.lmds_adapt_ts
    )

    counts, locs, labels, scores = lmds(pred)

    draw_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for p in locs[0]:
        y, x = int(p[0]), int(p[1])
        cv2.circle(draw_img, (x, y), 2, (0, 0, 255), -1)

    save_path = os.path.join(
        args.output_path,
        os.path.basename(args.img_path)
    )
    cv2.imwrite(save_path, draw_img)

    print(f"[INFO] Count: {counts[0]}")
    print(f"[INFO] Saved to: {save_path}")


if __name__ == '__main__':
    args = args_parser()

    vis(args)
