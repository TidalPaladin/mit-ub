#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mit_ub import __version__


def test_version():
    assert isinstance(__version__, str)
