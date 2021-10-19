#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import random

import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100

from lib.dataset.randaugment import RandAugmentMC

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def random_subset(dataset, n_samples, seed):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    return Subset(dataset, indices=indices[:n_samples])


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


TRANSFORM_LABELED_CIFAR10 = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ]
)

TRANSFORM_UNLABELED_CIFAR10 = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
TRANSFORM_TEST_CIFAR10 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=cifar10_mean, std=cifar10_std)]
)

TRANSFORM_LABELED_CIFAR100 = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ]
)

TRANSFORM_UNLABELED_CIFAR100 = TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
TRANSFORM_TEST_CIFAR100 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=cifar100_mean, std=cifar100_std)]
)


def get_cifar10(root: str, student_dataset_max_size: int, student_seed: int):
    labeled_dataset = CIFAR10(
        root=root, train=True, download=True, transform=TRANSFORM_LABELED_CIFAR10
    )
    test_dataset = CIFAR10(
        root=root, train=False, download=True, transform=TRANSFORM_TEST_CIFAR10
    )
    unlabeled_dataset = CIFAR10(
        root=root, train=True, download=True, transform=TRANSFORM_UNLABELED_CIFAR10
    )
    student_dataset = random_subset(
        dataset=CIFAR10(
            root=root, train=True, download=True, transform=TRANSFORM_LABELED_CIFAR10
        ),
        n_samples=student_dataset_max_size,
        seed=student_seed,
    )

    return {
        "labeled": labeled_dataset,
        "unlabeled": unlabeled_dataset,
        "test": test_dataset,
        "student": student_dataset,
    }


def get_cifar100(root: str, student_dataset_max_size: int, student_seed: int):
    labeled_dataset = CIFAR100(
        root=root, train=True, download=True, transform=TRANSFORM_LABELED_CIFAR100
    )
    test_dataset = CIFAR100(
        root=root, train=False, download=True, transform=TRANSFORM_TEST_CIFAR100
    )
    unlabeled_dataset = CIFAR100(
        root=root, train=True, download=True, transform=TRANSFORM_UNLABELED_CIFAR100
    )
    student_dataset = random_subset(
        dataset=CIFAR100(
            root=root, train=True, download=True, transform=TRANSFORM_LABELED_CIFAR100
        ),
        n_samples=student_dataset_max_size,
        seed=student_seed,
    )

    return {
        "labeled": labeled_dataset,
        "unlabeled": unlabeled_dataset,
        "test": test_dataset,
        "student": student_dataset,
    }
