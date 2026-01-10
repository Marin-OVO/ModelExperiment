import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from utils.lmds import LMDS
from .logger import *
from .averager import *
from .loss import *


# train/val epoch: train -> FocalLoss
#                  val   -> LMDS
def train_one_epoch(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        device: torch.device,
        logger: logging.Logger,
        print_freq: int,
        args,
        ) -> float:
    """
        Img   -> Model        -> dense map  (H×W)
        GT    -> PointsToMask -> dense mask (H×W)
        Loss  -> FocalLoss(map, mask)
    """
    # metric indicators
    first_order_loss = AverageMeter(20)
    second_order_loss = AverageMeter(20)
    losses = AverageMeter(20)
    batch_times = AverageMeter(20)

    # print freq 8 times for a epoch
    freq = len(train_dataloader) // print_freq
    print_freq_lst = [i * freq for i in range(1, 8)]
    print_freq_lst.append(len(train_dataloader) - 1)

    batch_start = time.time()

    model.train()

    lr = optimizer.param_groups[0]['lr']

    # FocalLoss
    criterion = FocalLoss(reduction="mean", normalize=True)
    for step, (images, targets) in enumerate(train_dataloader):

        images = images.to(device)

        # train results
        outputs = model(images)             # (B, 2, H, W) logits
        outputs = torch.sigmoid(outputs)    # Must heatmap -> FocalLoss()(Now UNet -> logits)

        gt_mask = targets.to(device).long() # (B, 2, H, W)

        # if gt_mask.dim() == 4:
        #     gt_mask = gt_mask.squeeze(1)

        # Loss1 -> UNet(pred loss)
        loss = criterion(outputs, gt_mask)

        first_order_loss.update(loss.item())
        second_order_loss.update(0.0)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in print_freq_lst:
            logger.info(
                "Epoch [{:^3}/{:<3}] | Iter {:^5} | LR {:.6f} | "
                "First {:^6.3f}({:^6.3f}) | "
                "Second {:.3f}({:.3f}) | "
                "Total {:^6.3f}({:^6.3f})".format(
                    epoch, args.epoch,
                    step, lr,
                    first_order_loss.val, first_order_loss.avg,
                    second_order_loss.val, second_order_loss.avg,
                    losses.val, losses.avg,
                )
            )

    out = losses.avg

    batch_end = time.time()
    batch_times.update(batch_end-batch_start)

    logger.info(
        "Val: Epoch [{:^3}/{:<3}] | "
        "First {:.3f}({:.3f}) | "
        "Second {:.3f}({:.3f}) | "
        "Total {:.3f}({:.3f}) | "
        "Time {:.2f}s".format(
            epoch, args.epoch,
            first_order_loss.val, first_order_loss.avg,
            second_order_loss.val, second_order_loss.avg,
            losses.val, losses.avg,
            batch_times.avg,
        )
    )

    return out, lr


@torch.no_grad
def val_one_epoch(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        metrics: object,
        args
        ) -> Union[float, torch.Tensor]:
    """
        Img    -> Model       -> dense map (H×W)
        Preds  -> LMDS        -> point list [(y, x), ...]
        GT     -> points list -> [(y, x), ...](N, 2)
        Metric -> mAP / recall / precision
    """
    metrics.flush()

    iter_metrics = metrics.copy()

    model.eval()

    for step, (images, targets) in enumerate(val_dataloader):
        images = images.cuda()

        # val results
        outputs = model(images)
        outputs = torch.sigmoid(outputs)

        points = targets['points']
        labels = targets['labels']

        # if step == 0:
        #     print(type(points))
        #     print(np.array(points).shape)

        # if isinstance(points, torch.Tensor):
        #     points = points.squeeze(0).tolist()

        if isinstance(labels, torch.Tensor):
            labels = labels.squeeze(0).tolist()

        points = np.asarray(points) # (1, N, 2)

        # (N, 2, 1) -> (N, 2)
        if points.ndim == 3 and points.shape[-1] == 1:
            points = points.squeeze(-1)

        # downsample (1, N, 2) -> (N, 2)
        if points.ndim == 3 and points.shape[0] == 1:
                points = points.squeeze(0)

        assert points.ndim == 2 and points.shape[1] == 2, \
            f"Invalid GT points shape: {points.shape}"

        gt_coords = [(int(p[1]), int(p[0])) for p in points]
        gt_labels = labels

        gt = dict(
            loc=gt_coords,
            labels=gt_labels
        )

        # int -> (tuple)
        ks = args.lmds_kernel_size
        if isinstance(ks, int):
            ks = (ks, ks)

        lmds = LMDS(
            kernel_size=ks,
            adapt_ts=args.lmds_adapt_ts
        )

        counts, locs, labels, scores = lmds(outputs)

        locs_pred = np.asarray(locs[0])
        if locs_pred.ndim == 3 and locs_pred.shape[-1] == 1:
            locs_pred = locs_pred.squeeze(-1)

        preds = dict(
            loc=locs_pred,
            labels=labels[0],
            scores=scores[0],
        )

        iter_metrics.feed(**dict(gt=gt, preds=preds))
        iter_metrics.aggregate()

        iter_metrics.flush()
        metrics.feed(**dict(gt=gt, preds=preds))

    mAP = np.mean([metrics.ap(c) for c in range(1, metrics.num_classes)]).item()

    metrics.aggregate()

    recall = metrics.recall()
    precision = metrics.precision()
    f1_score = metrics.fbeta_score()
    accuracy = metrics.accuracy()

    tmp_results = {
        'epoch': epoch,
        'f1_score': f1_score,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        "mAP": mAP
    }

    return tmp_results # a dict?