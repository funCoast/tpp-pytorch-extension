###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

from .._C import _xsmm as xsmm_cpp
from contextlib import contextmanager
from enum import IntEnum


class BrgemmBackend(IntEnum):
    LIBXSMM = 0
    SMELT = 1


def _normalize_backend(backend):
    if isinstance(backend, BrgemmBackend):
        return backend
    if isinstance(backend, str):
        text = backend.strip().lower()
        if text in ("libxsmm", "xsmm"):
            return BrgemmBackend.LIBXSMM
        if text in ("smelt", "sme"):
            return BrgemmBackend.SMELT
        raise ValueError(f"Unknown BrGEMM backend: {backend!r}")
    return BrgemmBackend(int(backend))


def manual_seed(seed):
    xsmm_cpp.manual_seed(seed)


def set_rng_state(new_state):
    raise NotImplementedError


def get_rng_state():
    raise NotImplementedError


def get_vnni_blocking(dtype):
    return xsmm_cpp.get_vnni_blocking(dtype)


def get_brgemm_backend():
    return BrgemmBackend(xsmm_cpp.get_brgemm_backend())


def set_brgemm_backend(backend):
    backend = _normalize_backend(backend)
    xsmm_cpp.set_brgemm_backend(int(backend))
    xsmm_cpp.set_smelt_auto_context_switch(backend == BrgemmBackend.SMELT)


@contextmanager
def brgemm_backend(backend):
    previous = get_brgemm_backend()
    set_brgemm_backend(backend)
    try:
        yield get_brgemm_backend()
    finally:
        set_brgemm_backend(previous)


# initialize libxsmm library and random number generator
xsmm_cpp.init_libxsmm()
manual_seed(12345)
