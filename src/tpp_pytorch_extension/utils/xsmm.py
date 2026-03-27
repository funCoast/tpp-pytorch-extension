###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import os
from .._C import _xsmm as xsmm_cpp
from contextlib import contextmanager
from enum import IntEnum


class BrgemmBackend(IntEnum):
    LIBXSMM = 0
    SMELT = 1


_BRGEMM_BACKEND_ENV = "TPP_BRGEMM_BACKEND"


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


def _backend_display_name(backend):
    backend = _normalize_backend(backend)
    return "SME-GEMM-dev" if backend == BrgemmBackend.SMELT else "libxsmm"


def _debug_print_backend(backend):
    print(f"[SME-GEMM-dev DEBUG]:当前使用的是 {_backend_display_name(backend)}")


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
    _debug_print_backend(backend)


@contextmanager
def brgemm_backend(backend):
    if backend is None:
        yield get_brgemm_backend()
        return
    previous = get_brgemm_backend()
    set_brgemm_backend(backend)
    try:
        yield get_brgemm_backend()
    finally:
        set_brgemm_backend(previous)


def _apply_brgemm_backend_from_env():
    backend = os.getenv(_BRGEMM_BACKEND_ENV)
    if backend is None or backend.strip() == "":
        _debug_print_backend(get_brgemm_backend())
        return
    set_brgemm_backend(backend)


# initialize libxsmm library and random number generator
xsmm_cpp.init_libxsmm()
manual_seed(12345)
_apply_brgemm_backend_from_env()
