#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os

import torch
from simple_parsing import ArgumentParser

from lib.dataset.cifar import get_cifar10, get_cifar100
from lib.pate.settings import PateCommonConfig
from lib.pate.utils import votes_aggregation_accuracy


def main(config_common: PateCommonConfig):
    result = []

    logging.info(
        f"Aggregating votes from {config_common.n_teachers} teachers in {config_common.model_dir} dir"
    )

    for teacher_id in range(config_common.n_teachers):
        votes = torch.load(
            os.path.join(config_common.model_dir, f"votes{teacher_id}.pt")
        )
        result.append(votes)

    agg_votes = sum(result)
    votes_path = os.path.join(config_common.model_dir, "aggregated_votes")
    torch.save(agg_votes, votes_path)

    logging.info(f"Saved votes ({agg_votes.shape}) to {votes_path}")

    if config_common.dataset == "cifar10":
        datasets = get_cifar10(
            root=config_common.dataset_dir,
            student_dataset_max_size=config_common.student_dataset_max_size,
            student_seed=config_common.seed + 100,
        )
    elif config_common.dataset == "cifar100":
        datasets = get_cifar100(
            root=config_common.dataset_dir,
            student_dataset_max_size=config_common.student_dataset_max_size,
            student_seed=config_common.seed + 100,
        )
    else:
        raise ValueError(f"Unexpected dataset: {config_common.dataset}")

    agg_accuracy = votes_aggregation_accuracy(agg_votes, datasets["student"])
    logging.info(f"Teacher ensemble aggregated accuracy: {agg_accuracy}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_arguments(PateCommonConfig, dest="common")

    args = parser.parse_args()

    main(args.common)
