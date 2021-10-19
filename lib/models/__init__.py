#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .resnet import resnet18
from .wide_resnet import wideresnet

__all__ = ["resnet18", "wideresnet"]
