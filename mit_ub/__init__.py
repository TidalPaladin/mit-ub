#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .model import AdaptiveViT, ViT


__version__ = importlib.metadata.version("mit-ub")
__all__ = ["ViT", "AdaptiveViT", "__version__"]
