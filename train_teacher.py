#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import random
from typing import Any

import numpy as np
import torch
from simple_parsing import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from lib.dataset.canary import fill_canaries
from lib.dataset.cifar import get_cifar10, get_cifar100
from lib.fixmatch import train
from lib.pate.settings import PateTeacherConfig, PateCommonConfig


def partition_dataset_indices(dataset_len, n_teachers, teacher_id, seed=None):
    random.seed(seed)

    teacher_data_size = dataset_len // n_teachers
    indices = list(range(dataset_len))
    random.shuffle(indices)

    result = indices[
        teacher_id * teacher_data_size : (teacher_id + 1) * teacher_data_size
    ]

    logging.info(
        f"Teacher {teacher_id} processing {len(result)} samples. "
        f"First index: {indices[0]}, last index: {indices[-1]}. "
        f"Range: [{teacher_id * teacher_data_size}:{(teacher_id + 1) * teacher_data_size}]"
    )

    return result


def _vote_one_teacher(
    model: nn.Module,
    student_dataset: Dataset,
    config_teacher: PateTeacherConfig,
    n_classes: int,
    device: Any,
):
    student_data_loader = DataLoader(
        student_dataset,
        batch_size=config_teacher.learning.batch_size,
    )

    r = torch.zeros(0, n_classes).to(device)

    with torch.no_grad():
        for data, _ in student_data_loader:
            data = data.to(device)
            output = model(data)
            binary_vote = torch.isclose(
                output, output.max(dim=1, keepdim=True).values
            ).double()

            r = torch.cat((r, binary_vote), 0)

    return r


def main(
    teacher_id: int,
    config_common: PateCommonConfig,
    config_teacher: PateTeacherConfig,
    device: str,
):
    logging.info(
        f"Training teacher {teacher_id} out of {config_common.n_teachers}. Dataset: {config_common.dataset}"
    )
    device = torch.device(device)
    assert teacher_id < config_common.n_teachers
    fixmatch_config = config_teacher.fixmatch

    if config_common.dataset == "cifar10":
        datasets = get_cifar10(
            root=config_common.dataset_dir,
            student_dataset_max_size=config_common.student_dataset_max_size,
            student_seed=config_common.seed + 100
            # we want different seeds for splitting data between teachers and for picking student subset
        )
        labeled_dataset, unlabeled_dataset = datasets["labeled"], datasets["unlabeled"]
        test_dataset, student_dataset = datasets["test"], datasets["student"]

        n_classes = 10
    elif config_common.dataset == "cifar100":
        datasets = get_cifar100(
            root=config_common.dataset_dir,
            student_dataset_max_size=config_common.student_dataset_max_size,
            student_seed=config_common.seed + 100
            # we want different seeds for splitting data between teachers and for picking student subset
        )
        labeled_dataset, unlabeled_dataset = datasets["labeled"], datasets["unlabeled"]
        test_dataset, student_dataset = datasets["test"], datasets["student"]

        n_classes = 100
    else:
        raise ValueError(f"Unexpected dataset: {config_common.dataset}")

    if config_common.canary_dataset is not None:
        logging.info(
            f"Injecting canaries. Seed: {config_common.canary_dataset.seed}, N:{config_common.canary_dataset.N}"
        )
        orig_label_sum = sum(labeled_dataset.targets)
        fill_canaries(
            dataset=labeled_dataset,
            num_classes=len(labeled_dataset.classes),
            N=config_common.canary_dataset.N,
            seed=config_common.canary_dataset.seed,
        )
        canary_label_sum = sum(labeled_dataset.targets)
        logging.info(
            f"Canaries injected. Label sum before: {orig_label_sum}, after: {canary_label_sum}"
        )

    labeled_indices = partition_dataset_indices(
        dataset_len=len(labeled_dataset),
        n_teachers=config_common.n_teachers,
        teacher_id=teacher_id,
        seed=config_common.seed,
    )

    labeled_dataset.data = labeled_dataset.data[labeled_indices]
    labeled_dataset.targets = np.array(labeled_dataset.targets)[labeled_indices]

    logging.info(f"Training teacher {teacher_id} with {len(labeled_dataset)} samples")

    checkpoint_path = os.path.join(config_common.model_dir, f"teacher_{teacher_id}.ckp")
    summary_writer = SummaryWriter(log_dir=config_common.tensorboard_log_dir)

    logging.info(
        f"Launching training. Tensorboard dir: {summary_writer.log_dir}. Checkpoint path: {checkpoint_path}"
    )

    model, acc, loss = train(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        test_dataset=test_dataset,
        fixmatch_config=fixmatch_config,
        learning_config=config_teacher.learning,
        device=device,
        n_classes=n_classes,
        writer=summary_writer,
        writer_tag="teacher",
        checkpoint_path=checkpoint_path,
    )

    logging.info(f"Finished training. Reported accuracy: {acc}")

    summary_writer.add_scalar("All teachers final accuracy", acc, teacher_id)

    r = _vote_one_teacher(
        model=model,
        student_dataset=student_dataset,
        config_teacher=config_teacher,
        n_classes=n_classes,
        device=device,
    )
    votes_path = os.path.join(config_common.model_dir, f"votes{teacher_id}.pt")

    torch.save(r, votes_path)
    logging.info(f"Finished voting. Votes shape: {r.shape}. Path: {votes_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()

    parser.add_argument("--teacher-id", type=int, required=True, help="Teacher id")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_arguments(PateCommonConfig, dest="common")
    parser.add_arguments(PateTeacherConfig, dest="teacher")

    args = parser.parse_args()
    main(
        teacher_id=args.teacher_id,
        config_common=args.common,
        config_teacher=args.teacher,
        device=args.device,
    )
