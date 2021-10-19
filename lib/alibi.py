#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Optional, List

import torch
import torch.nn.functional as f
import warnings
from opacus.privacy_analysis import compute_rdp, get_privacy_spent
from scipy.optimize import Bounds, LinearConstraint, minimize


EPS = 1e-10
ROOT2 = 2.0 ** 0.5


class RandomizedLabelPrivacy:
    def __init__(
        self,
        sigma: float,
        delta: float = 1e-10,
        mechanism: str = "Laplace",
        device: Any = None,
        seed: Optional[int] = None,
    ):
        r"""
        A privacy engine for randomizing labels.

        Arguments
            mechanism: type of the mechansim, for now either normal or laplacian
        """
        self.sigma = sigma
        self.delta = delta
        assert mechanism.lower() in ("gaussian", "laplace")
        self.isNormal = mechanism.lower() == "gaussian"  # else is laplace
        self.seed = (
            seed if seed is not None else (torch.randint(0, 255, (1,)).item())
        )  # this is not secure but ok for experiments
        self.device = device
        self.randomizer = torch.Generator(device) if self.sigma > 0 else None
        self.reset_randomizer()
        self.step: int = 0
        self.eps: float = float("inf")
        self.alphas: List[float] = [i / 10.0 for i in range(11, 1000)]
        self.alpha = float("inf")
        self.train()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def reset_randomizer(self):
        if self.randomizer is not None:
            self.randomizer.manual_seed(self.seed)

    def increase_budget(self, step: int = 1):
        if self.sigma <= 0 or step <= 0:
            return

        self.step += step
        if self.isNormal:
            rdps = compute_rdp(1.0, self.sigma / ROOT2, self.step, self.alphas)
            self.eps, self.alpha = get_privacy_spent(self.alphas, rdps, self.delta)
        else:
            if self.step > 1:
                warnings.warn(
                    "It is not optimal to use multiple steps with Laplace mechanism"
                )
                self.eps *= self.step
            else:
                self.eps = 2 * ROOT2 / self.sigma

    def noise(self, shape):
        if not self._train or self.randomizer is None:
            return None
        noise = torch.zeros(shape, device=self.device)
        if self.isNormal:
            noise.normal_(0, self.sigma, generator=self.randomizer)
        else:  # is Laplace
            tmp = noise.clone()
            noise.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            tmp.exponential_(ROOT2 / self.sigma, generator=self.randomizer)
            noise = noise - tmp
        return noise

    @property
    def privacy(self):
        return self.eps, self.alpha


class Ohm:
    def __init__(
        self, privacy_engine: RandomizedLabelPrivacy, post_process: str = "mapwithprior"
    ):
        """
        One Hot Mixer

        creates a noised one-hot version of the targets and returns
        the cross entropy loss w.r.t. the noised targets.

        Args:
            sigma: Normal distribution standard deviation to sample noise from,
                if 0 no noising happens and this becomes strictly equivalent
                to the normal cross entropy loss.
            post_process: mode for converting the noised output to proper
                probabilities, current supported modes are:
                MinMax, SoftMax, MinProjection, MAP, RandomizedResponse
                see `post_process` for more details.
        """
        self.mode = post_process.lower()
        assert self.mode in (
            "minmax",
            "softmax",
            "minprojection",
            "map",
            "mapwithprior",
            "randomizedresponse",
        ), self.mode
        self.privacy_engine = privacy_engine
        self.device = privacy_engine.device
        self.label_change = 0.0  # tracks the probability of label changes
        self.beta = 0.99  # is used to update label_change

    def post_process(self, in_vec: torch.Tensor, output: Optional[torch.Tensor] = None):
        """
        convert a given vector to a probability mass vector.

        Has five modes for now:
            MinMax: in -> (in - min(in)) / (max(in) - min(in)) -> in / sum(in)
            SoftMax: in -> e^in / sum(e^in)
            MinpProjection: returns closes point in the surface that defines all
                possible probability mass functions for a given set
                of classes
            MAP: returns the probability mass function that solve the MAP for a
                cross entroy loss.
            RandomizedResponse: sets the largest value to 1 and the rest to zero,
                this way either the original label is kept or some random label is
                assigned so this is equivalent to randomized response.
        """
        if self.mode == "minmax":
            return self._minmax(in_vec)
        elif self.mode == "minprojection":
            return self._duchi_projection(in_vec)
        elif self.mode == "softmax":
            return self._map_normal(in_vec)
        elif "map" in self.mode:
            assert not ("prior" in self.mode and output is None)
            prior = (
                f.softmax(output.detach(), dim=-1)
                if "prior" in self.mode and output is not None
                else None
            )
            return (
                self._map_normal(in_vec, self.privacy_engine.sigma, prior)
                if self.privacy_engine.isNormal
                else self._map_laplace(in_vec, self.privacy_engine.sigma / ROOT2, prior)
            )
        else:  # self.mode == "randomizedresponse"
            return self._select_max(in_vec)

    def soft_target(self, output: torch.Tensor, target: torch.Tensor):
        """
        convert to one had and noise.

        output is just used to create the vector of same size and on the
        same device.
        """
        if len(target.shape) == 1:  # targets are not soft labels
            target, is_noised = self._create_soft_target(output, target)
        else:
            is_noised = True  # TODO we assume that a soft target is alwasy noised
        if is_noised:
            target = self.post_process(target, output)

        return target

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        """
        calculates loss.

        Args:
            output: output of the network
            target: the labels (same dim 0 as the output)
        """
        pmf = self.soft_target(output, target)

        # calculate prob. of label change
        if len(target.shape) == 1:  # targets were not soft labels
            label = target.view(-1)
            maxidx = torch.argmax(pmf, dim=1).view(-1)
            lc = 1 - float((label == maxidx).sum().item()) / label.numel()
            self.label_change = self.label_change * self.beta + (1 - self.beta) * lc

        output = f.softmax(output, dim=-1)
        output = output + EPS
        output = -torch.log(output)
        return (pmf * output).sum(dim=-1).mean()

    def _create_soft_target(self, output: torch.Tensor, target: torch.Tensor):
        """
        Creates a soft target

        onehot representation if not noised and a noisy tensor if noised.
        returns a tuple of the soft target and whether it was noised
        """
        onehot = torch.zeros_like(output)
        target = target.type(torch.int64)
        onehot.scatter_(1, target.view(-1, 1), 1.0)
        rand = self.privacy_engine.noise(onehot.shape)
        noised = onehot if rand is None else onehot + rand
        return noised, rand is not None

    def _map_normal(
        self,
        in_vec: torch.Tensor,
        noise_sigma: float = 0,
        prior: Optional[torch.Tensor] = None,
    ):
        """
        The function is calculating the posterior {P(label=i | X) | i in C}
        through P(X | label = i) * P(label = i)

        For Gaussian mechanism boils down to be

        1 / Normalizer * e ^ (x_i / sigam) * P(label = i)
               =
        1 / Normalizer * e ^ (x_i / sigma + ln(P(label= i)))
        """
        in_vec = in_vec / (1.0 if noise_sigma <= 0 else noise_sigma)
        in_vec = in_vec + (
            0.0 if prior is None or noise_sigma <= EPS else torch.log(prior)
        )
        return f.softmax(in_vec, dim=-1)

    def _map_laplace(
        self,
        in_vec: torch.Tensor,
        noise_b: float = 0,
        prior: Optional[torch.Tensor] = None,
    ):
        """
        The function is calculating the posterior {P(label=i | X) | i in C}
        through P(X | label = i) * P(label = i)

        For Laplace mechanism boils down to be

        1 / Normalizer * e ^ (- sum_j |x_j - (j == i)| / b) * P(label = i)
               =
        1 / Normalizer * e ^ (- sum_j |x_j - (j == i)| / b + ln(P(label= i)))
        """
        n, c = in_vec.shape
        in_vec = in_vec.repeat(1, c).view(-1, c, c)
        in_vec = in_vec - torch.eye(c).to(device=in_vec.device).repeat(n, 1, 1)
        in_vec = -1 * in_vec.abs().sum(dim=-1)
        in_vec = in_vec / (1.0 if noise_b <= 0 else noise_b)
        in_vec = in_vec + (0.0 if prior is None or noise_b <= EPS else torch.log(prior))
        return f.softmax(in_vec, dim=-1)

    def _minmax(self, in_vec: torch.Tensor):
        """
        Converts `in_vec` in to a probability mass function.

        does this:
        in = (in - min(in)) / (max(in) - min(in))
        in = in / sum(in)
        """
        m = in_vec.min(-1)[0].reshape(-1, 1)
        M = in_vec.max(-1)[0].reshape(-1, 1)
        in_vec = (in_vec - m) / M
        return in_vec / in_vec.sum(-1).reshape(-1, 1)

    def _select_max(self, in_vec: torch.Tensor):
        """
        Implements randomized response.

        With prob x keeps the class with prob 1 - x assings random other class.
        """
        maxidx = torch.argmax(in_vec, dim=1).view(-1, 1).type(torch.int64)
        onehot = torch.zeros_like(in_vec)
        onehot.scatter_(1, maxidx, 1.0)
        return onehot

    def _minprojection_with_optimizer(self, in_vec: torch.Tensor):
        """
        Converts `in_vec` in to a probability mass function.

        minimizes the distance of in_vec to the plain `∑ x = 1`
        with constraints being `0 <= x <= 1`
        """
        n = in_vec.shape[-1]  # num classes
        bounds = Bounds([0] * n, [1] * n)  # all values in [0, 1]
        linear_constraint = LinearConstraint([[1] * n], [1], [1])  # values sum to 1
        x0 = [1 / n] * n  # initial point in the middle of the plain

        results = []

        class optim_wrapper:
            """
            wrapps optimiztion process for a single point
            """

            def __init__(self, p):
                self.p = p

            def func(self, x):
                return ((x - self.p) * (x - self.p)).sum()

            def jac(self, x):
                return 2 * (x - self.p)

            def hess(self, x):
                return 2 * torch.eye(n)

            def __call__(self):
                res = minimize(
                    self.func,
                    x0,
                    method="trust-constr",
                    jac=self.jac,
                    hess=self.hess,
                    constraints=linear_constraint,
                    bounds=bounds,
                )
                return res.x

        results = [optim_wrapper(x)() for x in in_vec.tolist()]
        return torch.Tensor(results).to(in_vec.device)

    def _minprojection_fast(self, in_vec: torch.Tensor):
        """
        Provides a much faster way to calculate  `_minprojection_with_optimizer`.

        This is our proposed method for calculating the min projection on a
        probability surface, i.e. ∑ x = 1 , 0 <= x <= 1

        The method completely bypasses an optimizer and instead uses recursion.
        The number of recursions in the stack is bounded by the number of classes.

        The method works as follows:
        given `p` is the input (it is a `k` element vector).

        1. Update `p` to its projection on `∑ x = 1`.
        2. Set negative elements of `p` to 0.
        3. Tracking the indices, reproject non-negative elemets of `p` to a `∑ x = 1` in the new space.
        4. Repeat from 2 until there are not negative elements.

        This algorithm is > 200 times faster on an average server
        """

        def fast_minimizer(vec):
            n = vec.shape[-1]
            point = vec + (1 - vec.sum().item()) / n
            if (point < 0).sum() > 0:
                idx = point >= 0
                point[point < 0] = 0
                p = point[idx]
                result = fast_minimizer(p)
                point[idx] = result
            return point

        results = [fast_minimizer(torch.Tensor(x)) for x in in_vec.tolist()]
        return torch.stack(results, 0).to(in_vec.device)

    def _minprojection_faster(self, in_vec: torch.Tensor):
        """
        This is a yet faster version of ``_minprojection_fast``.

        The code vectorizes the operation for a batch. Also converts
        recursion to a loop. This ads another >60 times speed up on
        top of ``_minprojection_fast``coming to about 4 orders of magnitude
        speed-up!
        """
        projection = in_vec.clone()
        n = projection.shape[-1]
        idx = torch.ones_like(projection) < 0  # all false
        for _ in range(n):
            projection[idx] = 0
            projection += (
                ((1 - projection.sum(-1)) / (n - idx.sum(-1))).view(-1, 1).repeat(1, n)
            )
            projection[idx] = 0
            idx_new = projection < 0
            if idx_new.sum().item() == 0:
                break
            else:
                idx = torch.logical_or(idx_new, idx)
        return projection

    def _duchi_projection(self, in_vec: torch.Tensor):
        """
        This implementation implements the procedure for projection
        onto a probabilistic simplex following the same notations as
        Algorithm 4 from our paper.
        """
        o = in_vec.clone().cpu()
        B, C = o.shape
        s, _ = torch.sort(o, dim=1, descending=True)
        cumsum = torch.cumsum(s, dim=1)
        indices = torch.arange(1, C + 1)
        k, _ = torch.max(
            (s * indices > (cumsum - 1)) * indices, dim=1
        )  # hack to get the last argmax
        u = (cumsum[torch.arange(B), k - 1] - 1) / k
        proj_o = (o - u.unsqueeze(1)).clamp(min=0)
        return proj_o


class NoisedCIFAR(torch.utils.data.Dataset):
    def __init__(
        self,
        cifar: torch.utils.data.Dataset,
        num_classes: int,
        randomized_label_privacy: RandomizedLabelPrivacy,
    ):
        self.cifar = cifar
        self.rlp = randomized_label_privacy
        targets = cifar.targets
        self.soft_targets = [self._noise(t, num_classes) for t in targets]
        self.rlp.increase_budget()  # increase budget
        # calculate probability of label change
        num_label_changes = sum(
            label != torch.argmax(soft_target).item()
            for label, soft_target in zip(targets, self.soft_targets)
        )
        self.label_change = num_label_changes / len(targets)

    def _noise(self, label, n):
        onehot = torch.zeros(n).to()
        onehot[label] = 1
        rand = self.rlp.noise((n,))
        return onehot if rand is None else onehot + rand

    def __len__(self):
        return self.cifar.__len__()

    def __getitem__(self, index):
        image, label = self.cifar.__getitem__(index)
        return image, self.soft_targets[index], label
