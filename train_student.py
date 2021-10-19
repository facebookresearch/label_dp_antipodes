#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import os

import numpy as np
import torch
from simple_parsing import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from lib.dataset.cifar import get_cifar10, get_cifar100
from lib.fixmatch import train
from lib.pate.accountant import run_analysis
from lib.pate.settings import PateStudentConfig, PateCommonConfig
from lib.pate.utils import (
    set_seed,
    noisy_threshold_labels,
    noisy_votes_aggregation_accuracy,
)


def get_eps(votes: torch.Tensor, student_config: PateStudentConfig):
    try:
        if (
            student_config.noise.result_noise == 0
            or student_config.noise.selection_noise == 0
        ):
            return math.inf

        eps_total, partition, answered, order_opt = run_analysis(
            votes.numpy(),
            "gnmax_conf",
            student_config.noise.result_noise,
            {
                "sigma1": student_config.noise.selection_noise,
                "t": student_config.noise.threshold,
            },
        )

        for i, x in enumerate(answered):
            if int(x) >= student_config.n_samples:
                return eps_total[i]

        return -1
    except Exception:
        return -1


def main(
    config_common: PateCommonConfig,
    config_student: PateStudentConfig,
    device: str,
):
    device = torch.device(device)
    fixmatch_config = config_student.fixmatch

    logging.info(
        f"Training student. {config_common.n_teachers} teachers, {config_common.dataset}"
    )

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

    set_seed(config_common.seed)
    votes_path = os.path.join(config_common.model_dir, "aggregated_votes")
    votes = torch.load(votes_path).cpu()

    student_indices = np.array(student_dataset.indices)
    labels, threshold_mask = noisy_threshold_labels(
        votes=votes,
        threshold=config_student.noise.threshold,
        selection_noise_scale=config_student.noise.selection_noise,
        result_noise_scale=config_student.noise.result_noise,
    )
    threshold_indices = threshold_mask.nonzero().numpy().squeeze()

    indices = student_indices[threshold_indices][: config_student.n_samples]
    labels = labels[: config_student.n_samples]

    labeled_dataset.data = labeled_dataset.data[indices]
    labeled_dataset.targets = labels

    noisy_agg_accuracy = noisy_votes_aggregation_accuracy(
        labeled_dataset, student_dataset, threshold_indices.squeeze()
    )
    logging.info(
        f"Added noise ("
        f"{config_student.noise.result_noise}, "
        f"{config_student.noise.selection_noise}, "
        f"{config_student.noise.threshold})"
    )
    logging.info(f"Noisy teacher enseble accuracy: {noisy_agg_accuracy}")
    eps = get_eps(votes, config_student)

    checkpoint_path = os.path.join(
        config_common.model_dir, f"{config_student.filename()}.ckp"
    )
    summary_writer = SummaryWriter(config_common.tensorboard_log_dir)

    logging.info(
        f"Launching training. Tensorboard dir: {summary_writer.log_dir}. Checkpoint path: {checkpoint_path}"
    )
    model, acc, loss = train(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        test_dataset=test_dataset,
        fixmatch_config=fixmatch_config,
        learning_config=config_student.learning,
        device=device,
        n_classes=n_classes,
        writer=summary_writer,
        writer_tag="student",
        checkpoint_path=checkpoint_path,
    )

    model_path = os.path.join(config_common.model_dir, config_student.filename())
    torch.save(model.state_dict(), model_path)

    logging.info(f"Finished training. Reported accuracy: {acc}, eps={eps}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_arguments(PateCommonConfig, dest="common")
    parser.add_arguments(PateStudentConfig, dest="student")

    args = parser.parse_args()
    main(config_common=args.common, config_student=args.student, device=args.device)
