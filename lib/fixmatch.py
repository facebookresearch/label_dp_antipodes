#!/usr/bin/env python3

"""
FixMatch training script
adapted from https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
under MIT LICENSE

MIT License

Copyright (c) 2019 Jungdae Kim, Qing Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import math
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from lib.models.wide_resnet import wideresnet
from lib.pate.settings import FixmatchConfig, LearningConfig, FixmatchModelConfig
from lib.pate.utils import AverageMeter, accuracy, build_optimizer


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_seed(seed):
    logging.debug(f"Setting global seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def create_model(config: FixmatchModelConfig, num_classes: int):
    return wideresnet(
        depth=config.depth,
        widen_factor=config.width,
        dropout=0,
        num_classes=num_classes,
    )


def test(device, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for (inputs, targets) in test_loader:
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    logging.info("top-1 acc: {:.2f}".format(top1.avg))
    logging.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def train(
    labeled_dataset: VisionDataset,
    unlabeled_dataset: VisionDataset,
    test_dataset: VisionDataset,
    fixmatch_config: FixmatchConfig,
    learning_config: LearningConfig,
    device: Any,
    n_classes: int,
    writer: SummaryWriter,
    writer_tag: str = "",
    checkpoint_path: Optional[str] = None,
):
    logging.info("Launching FixMatch training")

    batch_size = learning_config.batch_size
    steps_per_epoch = len(unlabeled_dataset) // batch_size

    num_expand_x = math.ceil(batch_size * steps_per_epoch / len(labeled_dataset))
    indices = np.hstack([np.arange(len(labeled_dataset)) for _ in range(num_expand_x)])

    labeled_dataset.data = labeled_dataset.data[indices]
    labeled_dataset.targets = np.array(labeled_dataset.targets)[indices]

    logging.info(f"Labeled expand coef: {num_expand_x}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Expanded labeled ds size: {len(labeled_dataset)}")

    labeled_trainloader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size * fixmatch_config.mu,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size,
    )

    logging.info(f"Labeled dataloader: {len(labeled_trainloader)}")
    logging.info(f"Unlabeled dataloader: {len(unlabeled_trainloader)}")
    logging.info(f"Test dataloader: {len(test_loader)}")

    model = create_model(fixmatch_config.model, num_classes=n_classes)
    model = model.to(device)

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": learning_config.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = build_optimizer(grouped_parameters, learning_config.optim)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        fixmatch_config.warmup,
        num_training_steps=len(labeled_trainloader) * learning_config.epochs,
    )

    ema_model = None
    if fixmatch_config.use_ema:
        from lib.models.ema import ModelEMA

        ema_model = ModelEMA(device, model, fixmatch_config.ema_decay)

    if fixmatch_config.amp:
        from apex import amp

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=fixmatch_config.opt_level
        )
    else:
        model = torch.nn.DataParallel(model)

    model.zero_grad()

    best_acc = 0
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    logging.info("starting training")

    for epoch in tqdm(range(learning_config.epochs)):
        model.train()
        logging.info(f"Epoch {epoch}. Memory {torch.cuda.memory_allocated(device)}")
        for _ in range(steps_per_epoch):

            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                logging.debug("Finished labeled trainloader")
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                logging.debug("Finished unlabeled trainloader")
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)),
                2 * fixmatch_config.mu + 1,
            ).to(device)
            targets_x = targets_x.to(device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * fixmatch_config.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(
                logits_u_w.detach() / fixmatch_config.T, dim=-1
            )
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(fixmatch_config.threshold).float()

            Lu = (
                F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask
            ).mean()

            loss = Lx + fixmatch_config.lambda_u * Lu

            if fixmatch_config.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

        if ema_model is not None:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test(device, test_loader, test_model)

        writer.add_scalar(f"train_{writer_tag}/1.train_loss", losses.avg, epoch)
        writer.add_scalar(f"train_{writer_tag}/2.train_loss_x", losses_x.avg, epoch)
        writer.add_scalar(f"train_{writer_tag}/3.train_loss_u", losses_u.avg, epoch)
        writer.add_scalar(f"train_{writer_tag}/4.mask", mask_probs.avg, epoch)
        writer.add_scalar(f"test_{writer_tag}/1.test_acc", test_acc, epoch)
        writer.add_scalar(f"test_{writer_tag}/2.test_loss", test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc

            if checkpoint_path:
                logging.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)

                if ema_model:
                    torch.save(test_model.state_dict(), checkpoint_path + "_ema")

        test_accs.append(test_acc)
        logging.info("Best top-1 acc: {:.2f}".format(best_acc))
        logging.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))

    final_model = ema_model.ema if ema_model else model
    return final_model, best_acc, test_loss
