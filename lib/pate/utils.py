#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Optional

import numpy as np
import torch
from lib.pate.settings import OptimizerConfig
from torch import optim
from torch.distributions import laplace, normal
from torch.utils.data import random_split, DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import VisionDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_aggregation_accuracy(noisy_dataset):
    correct = 0
    for i in range(len(noisy_dataset)):
        actual_label = noisy_dataset.dataset[i][1]
        noisy_label = noisy_dataset.labels[i]

        if noisy_label == actual_label:
            correct += 1

    return correct / len(noisy_dataset)


def votes_aggregation_accuracy(votes, dataset):
    correct = 0
    labels = votes.argmax(dim=1)
    for i in range(len(dataset)):
        actual_label = dataset[i][1]
        agg_label = labels[i]

        if agg_label == actual_label:
            correct += 1

    return correct / len(dataset)


def noisy_votes_aggregation_accuracy(noisy_dataset, dataset, indices):
    correct = 0
    labels = noisy_dataset.targets
    for i in range(len(noisy_dataset)):
        actual_label = dataset[indices[i]][1]
        agg_label = labels[i]

        if agg_label == actual_label:
            correct += 1

    return correct / len(noisy_dataset)


def convert_to_tensor_dataset(dataset: Dataset, n_samples: int):
    data = []
    labels = []

    for i in range(n_samples):
        datum, label = dataset[i]

        data.append(datum.numpy())
        labels.append(label)

    return TensorDataset(
        torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels))
    )


def convert_to_tensor_dataset_fixmatch_unlabeled(
    dataset: VisionDataset, n_samples: Optional[int] = None
):
    dataA = []
    dataB = []
    if n_samples is None:
        n_samples = len(dataset)

    for i in range(n_samples):
        (datumA, datumB), _ = dataset[i]

        dataA.append(datumA.numpy())
        dataB.append(datumA.numpy())

    return TensorDataset(
        torch.FloatTensor(np.array(dataA)), torch.FloatTensor(np.array(dataB))
    )


def partition_private_dataset(dataset, n_teachers, seed=None):
    teacher_data_size = len(dataset) // n_teachers
    lengths = [teacher_data_size] * n_teachers
    if sum(lengths) != len(dataset):
        lengths[-1] += len(dataset) - sum(lengths)

    return random_split(
        dataset, lengths=lengths, generator=torch.Generator().manual_seed(seed)
    )


def teachers_votes(student_dataset, teacher_models, batch_size, n_labels, device):
    student_data_loader = DataLoader(student_dataset, batch_size=batch_size)
    result = torch.zeros(len(teacher_models), len(student_dataset), n_labels)

    with torch.no_grad():
        for i, teacher in enumerate(teacher_models):
            r = torch.zeros(0, n_labels).to(device)
            for data, _ in student_data_loader:
                data = data.to(device)
                output = teacher(data)
                binary_vote = torch.isclose(
                    output, output.max(dim=1, keepdim=True).values
                ).double()

                r = torch.cat((r, binary_vote), 0)

            result[i] = r

    votes = result.sum(dim=0)
    return votes


def noisy_labels(votes, noise_scale):
    def noise(scale, shape):
        if scale == 0:
            return 0

        return laplace.Laplace(0, scale).sample(shape)

    noisy_votes = votes + noise(noise_scale, votes.shape)
    return noisy_votes.argmax(dim=1), noisy_votes


def noisy_threshold_labels(votes, threshold, selection_noise_scale, result_noise_scale):
    def noise(scale, shape):
        if scale == 0:
            return 0

        return normal.Normal(0, scale).sample(shape)

    noisy_votes = votes + noise(selection_noise_scale, votes.shape)

    over_t_mask = noisy_votes.max(dim=1).values > threshold
    over_t_labels = (
        votes[over_t_mask] + noise(result_noise_scale, votes[over_t_mask].shape)
    ).argmax(dim=1)

    return over_t_labels, over_t_mask


def build_optimizer(parameters, optimizer_config: OptimizerConfig):
    if optimizer_config.method == "SGD":
        optimizer = optim.SGD(
            parameters,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=optimizer_config.nesterov,
        )
    elif optimizer_config.method == "RMSprop":
        optimizer = optim.RMSprop(parameters, lr=optimizer_config.lr)
    elif optimizer_config.method == "Adam":
        optimizer = optim.Adam(
            parameters, lr=optimizer_config.lr, betas=(optimizer_config.momentum, 0.999)
        )
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    return optimizer


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5:  # warm-up
        lr = lr * float(epoch + 1) / 5
    elif epoch >= 150 and epoch < 250:
        lr *= 0.1
    elif epoch >= 250:
        lr *= 0.01 * (0.1 ** ((epoch - 250) // 100))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
