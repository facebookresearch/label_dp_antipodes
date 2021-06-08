#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates.

from .resnet import resnet18
from .wide_resnet import wideresnet

__all__ = ["resnet18", "wideresnet"]
