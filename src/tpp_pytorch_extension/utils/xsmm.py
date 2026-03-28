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
import time
from contextlib import contextmanager
from enum import IntEnum

import torch
from .._C import _xsmm as xsmm_cpp


class BrgemmBackend(IntEnum):
    LIBXSMM = 0
    SMELT = 1


_BRGEMM_BACKEND_ENV = "TPP_BRGEMM_BACKEND"
_BRGEMM_COMPARE_ENV = "TPP_BRGEMM_COMPARE"


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


def _env_truthy(name):
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


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


def is_smelt_backend(backend=None):
    if backend is None:
        backend = get_brgemm_backend()
    return _normalize_backend(backend) == BrgemmBackend.SMELT


def should_compare_brgemm():
    return _env_truthy(_BRGEMM_COMPARE_ENV)


def describe_tensor(tensor):
    if tensor is None:
        return "None"
    if not torch.is_tensor(tensor):
        return f"{type(tensor).__name__}({tensor!r})"
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def log_brgemm_shapes(tag, named_tensors):
    print(f"[SME-GEMM-dev DEBUG]:{tag} shapes:")
    for name, tensor in named_tensors:
        print(f"  - {name}: {describe_tensor(tensor)}")


def log_brgemm_output_compare(tag, smelt_output, ref_output):
    print(f"[SME-GEMM-dev DEBUG]:{tag} output compare:")
    print(f"  - smelt: {describe_tensor(smelt_output)}")
    print(f"  - libxsmm: {describe_tensor(ref_output)}")
    if not torch.is_tensor(smelt_output) or not torch.is_tensor(ref_output):
        print("  - compare skipped: outputs are not tensors")
        return
    if smelt_output.shape != ref_output.shape:
        print(
            f"  - shape mismatch: smelt={tuple(smelt_output.shape)}, "
            f"libxsmm={tuple(ref_output.shape)}"
        )
        return

    smelt_finite = torch.isfinite(smelt_output)
    ref_finite = torch.isfinite(ref_output)
    shared_finite = smelt_finite & ref_finite
    total = smelt_output.numel()
    shared_count = int(shared_finite.sum().item())
    smelt_nan = int(torch.isnan(smelt_output).sum().item())
    ref_nan = int(torch.isnan(ref_output).sum().item())
    smelt_inf = int(torch.isinf(smelt_output).sum().item())
    ref_inf = int(torch.isinf(ref_output).sum().item())

    print(f"  - shared finite elements: {shared_count}/{total}")
    print(f"  - smelt nan/inf: {smelt_nan}/{smelt_inf}")
    print(f"  - libxsmm nan/inf: {ref_nan}/{ref_inf}")

    if shared_count == 0:
        print("  - no shared finite elements to compare")
        return

    abs_diff = (smelt_output[shared_finite] - ref_output[shared_finite]).abs()
    print(f"  - max abs diff: {abs_diff.max().item():.6e}")
    print(f"  - mean abs diff: {abs_diff.mean().item():.6e}")


def log_brgemm_timing_compare(tag, smelt_seconds, ref_seconds):
    smelt_ms = smelt_seconds * 1000.0
    ref_ms = ref_seconds * 1000.0
    delta_ms = smelt_ms - ref_ms
    if ref_ms > 0:
        pct = (delta_ms / ref_ms) * 100.0
    else:
        pct = float("inf") if delta_ms > 0 else 0.0
    if delta_ms < 0:
        verdict = "faster"
    elif delta_ms > 0:
        verdict = "slower"
    else:
        verdict = "same"
    print(f"[SME-GEMM-dev DEBUG]:{tag} timing compare:")
    print(f"  - smelt: {smelt_ms:.3f} ms")
    print(f"  - libxsmm: {ref_ms:.3f} ms")
    print(f"  - delta: {delta_ms:+.3f} ms ({pct:+.2f}%), smelt is {verdict}")


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
