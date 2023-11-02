#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib.metadata

from .model import BACKBONES


__version__ = importlib.metadata.version("mit-ub")
__all__ = ["BACKBONES"]
