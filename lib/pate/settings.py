#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True, frozen=True)
class PateNoiseConfig:
    selection_noise: float
    result_noise: float
    threshold: int


@dataclass(eq=True, frozen=True)
class CanaryDatasetConfig:
    N: int
    seed: int = 11337


@dataclass(eq=True, frozen=True)
class OptimizerConfig:
    method: str = "SGD"
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True


@dataclass(eq=True, frozen=True)
class LearningConfig:
    optim: OptimizerConfig = OptimizerConfig()
    batch_size: int = 64
    epochs: int = 40


@dataclass(eq=True, frozen=True)
class FixmatchModelConfig:
    width: int = 4
    depth: int = 28
    cardinality: int = 4


@dataclass(eq=True, frozen=True)
class FixmatchConfig:
    model: FixmatchModelConfig = FixmatchModelConfig()
    mu: int = 7
    warmup: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.999
    amp: bool = False
    opt_level: str = "O1"
    T: float = 1.0
    threshold: float = 0.95
    lambda_u: float = 1.0


@dataclass(eq=True, frozen=True)
class PateCommonConfig:
    n_teachers: int
    dataset: str = "cifar10"
    dataset_root: str = "/tmp/cifar10"
    seed: int = 1337
    student_dataset_max_size: int = 10000
    model_dir: str = "/tmp/pate"
    canary_dataset: Optional[CanaryDatasetConfig] = None
    tensorboard_log_dir: Optional[str] = None

    @property
    def dataset_dir(self):
        return os.path.join(self.dataset_root, self.dataset)


@dataclass(eq=True, frozen=True)
class PateStudentConfig:
    noise: PateNoiseConfig
    fixmatch: FixmatchConfig
    n_samples: int = 1000
    learning: LearningConfig = LearningConfig()

    def filename(self):
        noise = self.noise
        setting_str = (
            f"student_noise_{noise.result_noise:.0f}_{noise.selection_noise:.0f}_{noise.threshold}_"
            f"samples_{self.n_samples}_"
            f"epochs_{self.learning.epochs}"
        )

        setting_str += f"_{hash(self):x}"

        return setting_str


@dataclass(eq=True, frozen=True)
class PateTeacherConfig:
    learning: LearningConfig = LearningConfig()
    fixmatch: FixmatchConfig = FixmatchConfig()
