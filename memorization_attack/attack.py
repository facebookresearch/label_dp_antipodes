#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2021 Florian Tramer

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

import os
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100


def rand_pos_and_labels(trainset, N, seed=None):
    """
    Selects `N` random positions in the training set and `N` corresponding
    random incorrect labels.
    """
    np.random.seed(seed)
    num_classes = len(trainset.classes)
    rand_positions = np.random.choice(len(trainset.data), N, replace=False)
    rand_labels = []
    for idx in rand_positions:
        y = trainset.targets[idx]
        new_y = np.random.choice(list(set(range(num_classes)) - {y}))
        rand_labels.append(new_y)

    return rand_positions, rand_labels


def eps(acc):
    """
    Point estimate of epsilon-DP given the adversary's guessing accuracy
    """
    if acc <= 0.5:
        return 0
    if acc == 1:
        return np.inf
    return np.log(acc / (1 - acc))


canary_seed = 11337
canary_N = 1000

for data in ["cifar10", "cifar100"]:
    for algo in ["PATE", "RR", "RR_gaussian"]:
        for acc in ["HH", "H", "M", "L"]:

            if not os.path.exists(f"confs/confs_{data}_{algo}_{acc}.npy"):
                continue

            if data == "cifar10":
                train_dataset = CIFAR10(root="/tmp/cifar10", train=True, download=False)
            else:
                train_dataset = CIFAR100(root="/tmp/cifar100", train=True, download=False)

            # re-generate the canaries
            canary_positions, canary_labels = rand_pos_and_labels(
                train_dataset, N=canary_N, seed=canary_seed
            )
            if data == "cifar10":
                assert (np.sum(canary_positions), np.sum(canary_labels)) == (
                    25165634,
                    4641,
                )
            else:
                assert (np.sum(canary_positions), np.sum(canary_labels)) == (
                    25165634,
                    50213,
                )

            # model confidence on the canaries
            confs_on_canary = np.load(f"confs/confs_{data}_{algo}_{acc}.npy")
            c1 = confs_on_canary[np.arange(1000), canary_labels]
            c1 = c1.reshape(-1, 1)

            # build a matrix of size (1000, C-2) of model confidences in all other incorrect classes
            true_labels = np.asarray(train_dataset.targets)[canary_positions]
            incorrect_labels = [
                sorted(list(set(range(len(train_dataset.classes))) - {y1, y2}))
                for (y1, y2) in zip(true_labels, canary_labels)
            ]
            incorrect_labels = np.asarray(incorrect_labels)

            c2 = []
            for i in range(incorrect_labels.shape[1]):
                c2.append(confs_on_canary[np.arange(1000), incorrect_labels[:, i]])
            c2 = np.stack(c2, axis=-1)

            for i in range(1000):
                assert true_labels[i] not in incorrect_labels[i], i
                assert canary_labels[i] not in incorrect_labels[i], i

            # the adversary's guess: given a pair of labels (y', y''), the adversary
            # guesses that the label with higher confidence is the canary
            guesses = c1 > c2

            epsilons_low = []
            epsilons_high = []
            N = len(c1)

            # threshold on the max confidence of the two labels: if neither label has high enough confidence, abstain
            thresholds = [0.99]

            for threshold in thresholds:

                # abstain from guessing if confidence is too low
                dont_know = np.maximum(c1, c2) < threshold

                # number of guesses per canary
                weights = np.sum(1 - dont_know.astype(np.float32), axis=-1)

                # guessing acc per canary (when adv doesn't abstain)
                accs = np.sum(guesses * (1 - dont_know.astype(np.float32)), axis=-1)
                accs /= np.maximum(
                    np.sum((1 - dont_know.astype(np.float32)), axis=-1), 1
                )

                # only consider canaries where we made at least one guess
                accs = accs[weights > 0]
                weights = weights[weights > 0]
                weights /= np.sum(weights)
                N_effective = len(accs)

                if N_effective:
                    # weighted mean
                    acc_mean = np.sum(accs * weights)

                    # weighted std: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
                    V = (
                        np.sum(weights * (accs - acc_mean) ** 2)
                        * 1
                        / (1 - np.sum(weights ** 2) + 1e-8)
                    )
                    acc_std = np.sqrt(V / N_effective)

                    # invert the guess if it's the wrong way around (the DP def. is symmetric so this is fine)
                    acc_mean = max(acc_mean, 1 - acc_mean)

                    # normal CI
                    acc_low = acc_mean - 1.96 * acc_std
                    acc_high = acc_mean + 1.96 * acc_std

                    # correction
                    acc_low = max(0.5, acc_low)
                    acc_high = min(1, acc_high)

                    # if all the guesses are correct, treat the accuracy as a Binomial with empirical probability of 1.0
                    if acc_mean == 1:
                        acc_low = min(acc_low, 1 - 3 / N_effective)
                else:
                    acc_low, acc_mean, acc_high = 0.0, 0.5, 1.0

                # epsilon CI
                e_low, e_high = eps(acc_low), eps(acc_high)
                epsilons_low.append(e_low)
                epsilons_high.append(e_high)
                # print(f"\t{threshold}, DK={np.mean(dont_know):.2f}, N={int(N_effective)}, acc={acc_mean:.2f}, eps=({e_low:.2f}, {e_high:.2f})")

            # if multiple thresholds were considered, select the best one
            # (this should be combined with a correction for multiple hypothesis testing)
            eps_low, eps_high = 0, 0
            best_threshold = None

            for el, eh, t in zip(epsilons_low, epsilons_high, thresholds):
                if el > eps_low:
                    eps_low = el
                    eps_high = eh
                    best_threshold = t
                elif (el == eps_low) & (eh > eps_high):
                    eps_high = eh
                    best_threshold = t
            print(f"{data}-{algo}-{acc}: ({eps_low:.1f}, {eps_high:.1f})")
        print()
