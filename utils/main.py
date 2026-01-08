import time

from utils.lmds import LMDS
from typing import Optional, Union
from .logger import *
from .averager import *

import torch
import torch.nn as nn

import numpy as np


def train_one_epoch(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        device: torch.device,
        logger: logging.Logger,
        print_freq: int,
        args
        ) -> float:
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
    for step, (images, targets) in enumerate(train_dataloader):

        images = images.to(device)

        if isinstance(targets, (list, tuple)):
            targets = [tar.to(device) for tar in targets]
        else:
            targets = targets.to(device)

        loss_dict = model(images, targets)
        first_order_loss.update(loss_dict['first_order_loss'].item())
        second_order_loss.update(loss_dict['second_order_loss'].item())

        loss = sum(loss for loss in loss_dict.values())
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in print_freq_lst:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "First:{first_order_loss.val:.3f}({first_order_loss.avg:.3f})  "
                "Second:{second_order_loss.val:.3f}({second_order_loss.avg:.3f})  "
                "Loss:{losses.val:.3f}({losses.avg:.3f})  ".format(
                    args.epoch, epoch, step,
                    first_order_loss=first_order_loss,
                    second_order_loss=second_order_loss,
                    losses=losses,
                )
            )

    out = losses.avg

    batch_end = time.time()
    batch_times.update(batch_end-batch_start)

    logger.info(
        "Epoch [{}].  "
        "First:{first_order_loss.val:.3f}({first_order_loss.avg:.3f})  "
        "Second:{second_order_loss.val:.3f}({second_order_loss.avg:.3f})  "
        "Loss:{losses.val:.3f}({losses.avg:.3f})  "
        "Time:{batch_times.avg:.2f}  ".format(
            args.epoch,
            first_order_loss=first_order_loss,
            second_order_loss=second_order_loss,
            losses=losses,
            batch_times=batch_times
        ))

    return out


@torch.no_grad
def val_one_epoch(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        metrics: object,
        args
        ) -> Union[float, torch.Tensor]:
    metrics.flush()

    iter_metrics = metrics.copy()

    model.eval()

    for step, (images, targets) in enumerate(val_dataloader):
        images = images.cuda()
        output, _ = model(images)

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()

        gt = dict(
            loc=gt_coords,
            labels=gt_labels
        )

        lmds = LMDS(
            kernel_size=args.lmds_kernel_size,
            adapt_ts=args.lmds_adapt_ts
        )

        counts, locs, labels, scores = lmds(output)

        preds = dict(
            loc=locs[0],
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

    return tmp_results