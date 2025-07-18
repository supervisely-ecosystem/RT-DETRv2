"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..data.transforms.masking import Masking, apply_strong_transform
from ..zoo.rtdetr import RTDETRPostProcessor

from supervisely.nn.training import train_logger


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    cfg = kwargs.get('cfg', None)
    postprocessor : RTDETRPostProcessor = cfg.postprocessor

    masking :Masking = kwargs.get('masking', None)

    for i, (samples, targets, unlabeled_samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        unlabeled_samples = unlabeled_samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        # with torch.no_grad():
            # teacher_pseudo_labels = model_teacher(unlabeled_samples)

        # 1. L_S loss
        # outputs = model(samples, targets=targets)
        # loss_dict_L_S = criterion(outputs, targets, **metas)

        # 2. L_T loss
        # ulabeled_samples_augmented = augs(unlabeled_samples)
        # outputs_student = model(ulabeled_samples_augmented, targets=teacher_pseudo_labels)
        # loss_dict_L_T = criterion2(outputs_student, teacher_pseudo_labels)

        # 3. L_M loss
        # masked_samples = masking(unlabeled_samples)
        # teacher_pseudo_labels_masking = filter_by_confidence(teacher_pseudo_labels, threshold=0.8)
        # apply_quality_weights
        # outputs_masked = model(masked_samples, targets=teacher_pseudo_labels_masking)
        # loss_dict_L_M = criterion3(outputs_masked, teacher_pseudo_labels_masking)

        # apply loss weights
        # loss_dict = {**loss_dict_L_S * 1.0,
        #              **loss_dict_L_T * LAMBDA_T,
        #              **loss_dict_L_M * LABDA_M}
        
        # loss : torch.Tensor = sum(loss_dict.values())
        # update steps
        # ...
        # ema.update()

        from rtdetrv2_pytorch.tools.utils import draw_tensor_grid
        draw_tensor_grid(samples[:4]).save("output/samples.png")
        draw_tensor_grid(unlabeled_samples[:4]).save("output/unlabeled_samples.png")

        loss_dict = {}

        optimizer.zero_grad()

        # 0. Teacher inference
        with torch.autocast(device_type="cuda"):
            teacher_outputs = ema.forward(unlabeled_samples)
            teacher_targets = postprocessor.postprocess(teacher_outputs)
            # TODO: handle filtering
            teacher_targets_filtered = filter_by_confidence(teacher_targets, threshold=0.01)
        del teacher_outputs

        # 1. L_S loss
        with torch.autocast(device_type="cuda"):
            outputs = model(samples, targets=targets)
            loss_dict_L_S = criterion(outputs, targets, **metas)
            loss_S = sum(loss_dict_L_S.values())
            loss_dict["L_S"] = loss_S.item()
        scaler.scale(loss_S).backward()
        del outputs, loss_dict_L_S, loss_S
        # torch.cuda.empty_cache()

        # 2. L_T loss
        with torch.autocast(device_type="cuda"):
            unlabeled_samples_augmented = apply_strong_transform(unlabeled_samples)
            draw_tensor_grid(unlabeled_samples_augmented[:4]).save("output/unlabeled_samples_augmented.png")
            outputs_student = model(unlabeled_samples_augmented, with_cdn=False)
            loss_dict_L_T = criterion(outputs_student, teacher_targets, **metas)
            loss_T = sum(loss_dict_L_T.values()) * 1.0
            loss_dict["L_T"] = loss_T.item()
        scaler.scale(loss_T).backward()
        del unlabeled_samples_augmented, outputs_student, loss_dict_L_T, loss_T
        # torch.cuda.empty_cache()

        # 3. L_M loss
        with torch.autocast(device_type="cuda"):
            masked_samples = masking(unlabeled_samples)
            draw_tensor_grid(masked_samples[:4]).save("output/masked_samples.png")
            outputs_student_M = model(masked_samples, with_cdn=False)
            # TODO: apply_quality_weights
            loss_dict_L_M = criterion(outputs_student_M, teacher_targets, **metas)
            loss_M = sum(loss_dict_L_M.values()) * 1.0
            loss_dict["L_M"] = loss_M.item()
        scaler.scale(loss_M).backward()
        del masked_samples, outputs_student_M, loss_dict_L_M, loss_M
        # torch.cuda.empty_cache()

        # # Combine losses and apply loss weights
        # loss : torch.Tensor = sum(loss_dict_L_S.values()) + \
        #     sum(loss_dict_L_T.values()) * 1.0 + \
        #     sum(loss_dict_L_M.values()) * 1.0

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()        
        
        # EMA update
        ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v, global_step)

        train_logger.step_finished()
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def filter_by_confidence(targets, threshold):
    """Filter targets by confidence threshold."""
    targets_new = []
    for target in targets:
        target_new = {}
        mask = target['scores'] > threshold
        target_new['boxes'] = target['boxes'][mask]
        target_new['labels'] = target['labels'][mask]
        target_new['scores'] = target['scores'][mask]
        targets_new.append(target_new)
    return targets_new
