#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np


def _rand_pos_and_labels(dataset, num_classes, N, seed=None):
    np.random.seed(seed)
    rand_positions = np.random.choice(len(dataset.data), N, replace=False)
    rand_labels = []
    for idx in rand_positions:
        y = dataset.targets[idx]
        new_y = np.random.choice(list(set(range(num_classes)) - {y}))
        rand_labels.append(new_y)
    return rand_positions, rand_labels


def fill_canaries(dataset, num_classes, N=1000, seed=None):
    """
    Returns the dataset, where `N` random points are assigned a random incorrect label.
    """
    rand_positions, rand_labels = _rand_pos_and_labels(
        dataset, num_classes, N, seed=seed
    )

    rand_positions = np.asarray(rand_positions)
    rand_labels = np.asarray(rand_labels)

    targets = np.asarray(dataset.targets)
    targets[rand_positions] = rand_labels

    dataset.targets = list(targets)

    return dataset
