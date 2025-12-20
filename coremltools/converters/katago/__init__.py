# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
KataGo model converter for Core ML.

This module provides functionality to convert KataGo neural network models
to Core ML format.
"""

__all__ = ["convert"]


def __getattr__(name):
    """Lazy import to avoid circular dependency during coremltools initialization."""
    if name == "convert":
        from ._converter import convert
        return convert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
