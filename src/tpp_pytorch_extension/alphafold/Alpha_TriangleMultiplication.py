###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################


import math
import time
import torch
from torch import nn
from torch.autograd import Function

from tpp_pytorch_extension._C import (
    _alpha_attention as Alpha_TriangleMultiplication_cpp,
)
from tpp_pytorch_extension.utils.xsmm import (
    BrgemmBackend,
    brgemm_backend,
    get_brgemm_backend,
    is_smelt_backend,
    log_brgemm_output_compare,
    log_brgemm_params,
    log_brgemm_timing_compare,
    should_compare_brgemm,
)


class TriangleMultiplicationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        mask,
        c_equation,
        layer_norm_input_weight,
        layer_norm_input_bias,
        left_projection_weight,
        left_projection_bias,
        right_projection_weight,
        right_projection_bias,
        left_gate_weight,
        left_gate_bias,
        right_gate_weight,
        right_gate_bias,
        center_layer_norm_weight,
        center_layer_norm_bias,
        output_projection_weight,
        output_projection_bias,
        gating_linear_weight,
        gating_linear_bias,
    ):
        equation_flag = int(0)
        if c_equation == "ikc,jkc->ijc":  # "Outgoing" edges equation
            equation_flag = 0
        else:  # "Incoming" edges equation
            equation_flag = 1
        act = Alpha_TriangleMultiplication_cpp.trianglemulti_forward(
            act.contiguous(),
            mask.contiguous(),
            equation_flag,
            layer_norm_input_weight,
            layer_norm_input_bias,
            left_projection_weight,
            left_projection_bias,
            right_projection_weight,
            right_projection_bias,
            left_gate_weight,
            left_gate_bias,
            right_gate_weight,
            right_gate_bias,
            center_layer_norm_weight,
            center_layer_norm_bias,
            output_projection_weight,
            output_projection_bias,
            gating_linear_weight,
            gating_linear_bias,
        )
        return act


def _run_triangle_multiplication_impl(
    act,
    mask,
    c_equation,
    layer_norm_input_weight,
    layer_norm_input_bias,
    left_projection_weight,
    left_projection_bias,
    right_projection_weight,
    right_projection_bias,
    left_gate_weight,
    left_gate_bias,
    right_gate_weight,
    right_gate_bias,
    center_layer_norm_weight,
    center_layer_norm_bias,
    output_projection_weight,
    output_projection_bias,
    gating_linear_weight,
    gating_linear_bias,
):
    if (
        act.dtype == torch.float16
        or mask.dtype == torch.float16
        or layer_norm_input_weight.dtype == torch.float16
        or layer_norm_input_bias.dtype == torch.float16
        or left_projection_weight.dtype == torch.float16
        or left_projection_bias.dtype == torch.float16
        or right_projection_weight.dtype == torch.float16
        or right_projection_bias.dtype == torch.float16
        or left_gate_weight.dtype == torch.float16
        or left_gate_bias.dtype == torch.float16
        or right_gate_weight.dtype == torch.float16
        or right_gate_bias.dtype == torch.float16
        or center_layer_norm_weight.dtype == torch.float16
        or center_layer_norm_bias.dtype == torch.float16
        or output_projection_weight.dtype == torch.float16
        or output_projection_bias.dtype == torch.float16
        or gating_linear_weight.dtype == torch.float16
        or gating_linear_bias.dtype == torch.float16
    ):
        return TriangleMultiplicationFunction.apply(
            act.to(torch.float16),
            mask.to(torch.float32),
            c_equation,
            layer_norm_input_weight.to(torch.float16),
            layer_norm_input_bias.to(torch.float16),
            left_projection_weight.to(torch.float16),
            left_projection_bias.to(torch.float32),
            right_projection_weight.to(torch.float16),
            right_projection_bias.to(torch.float32),
            left_gate_weight.to(torch.float16),
            left_gate_bias.to(torch.float32),
            right_gate_weight.to(torch.float16),
            right_gate_bias.to(torch.float32),
            center_layer_norm_weight.to(torch.float16),
            center_layer_norm_bias.to(torch.float16),
            output_projection_weight.to(torch.float16),
            output_projection_bias.to(torch.float32),
            gating_linear_weight.to(torch.float16),
            gating_linear_bias.to(torch.float32),
        )

    return TriangleMultiplicationFunction.apply(
        act,
        mask,
        c_equation,
        layer_norm_input_weight,
        layer_norm_input_bias,
        left_projection_weight,
        left_projection_bias,
        right_projection_weight,
        right_projection_bias,
        left_gate_weight,
        left_gate_bias,
        right_gate_weight,
        right_gate_bias,
        center_layer_norm_weight,
        center_layer_norm_bias,
        output_projection_weight,
        output_projection_bias,
        gating_linear_weight,
        gating_linear_bias,
    )


def TriangleMultiplicationOpti_forward(self, act, mask, backend=None):
    mask = mask[..., None]
    effective_backend = get_brgemm_backend() if backend is None else backend
    compare_active = should_compare_brgemm() and is_smelt_backend(effective_backend)
    with brgemm_backend(effective_backend, verbose=not compare_active):
        input_act = act
        if compare_active:
            compare_start = time.perf_counter()
        smelt_act = _run_triangle_multiplication_impl(
            input_act,
            mask,
            self.c_equation,
            self.layer_norm_input.weight,
            self.layer_norm_input.bias,
            self.left_projection.weight,
            self.left_projection.bias,
            self.right_projection.weight,
            self.right_projection.bias,
            self.left_gate.weight,
            self.left_gate.bias,
            self.right_gate.weight,
            self.right_gate.bias,
            self.center_layer_norm.weight,
            self.center_layer_norm.bias,
            self.output_projection.weight,
            self.output_projection.bias,
            self.gating_linear.weight,
            self.gating_linear.bias,
        )
        if compare_active:
            smelt_elapsed = time.perf_counter() - compare_start

        if compare_active:
            act_dim = int(self.left_projection.in_features)
            num_intermediate_channel = int(self.left_projection.out_features)
            tri_blocksize = 32
            b = int(input_act.shape[0])
            s = int(input_act.shape[1])
            b_pad = ((b + tri_blocksize - 1) // tri_blocksize) * tri_blocksize
            s_pad = ((s + tri_blocksize - 1) // tri_blocksize) * tri_blocksize
            log_brgemm_params(
                "TriangleMultiplication",
                [
                    (
                        "proj_brgemm",
                        {
                            "M": tri_blocksize,
                            "N": num_intermediate_channel,
                            "K": act_dim,
                            "lda": act_dim,
                            "ldb": num_intermediate_channel,
                            "ldc": num_intermediate_channel,
                            "count": 1,
                            "beta": 0.0,
                            "a_trans": 0,
                            "b_vnni": 1,
                            "transa": "N",
                            "transb": "N",
                        },
                    ),
                    (
                        "equation_brgemm_outgoing",
                        {
                            "M": tri_blocksize,
                            "N": tri_blocksize,
                            "K": tri_blocksize,
                            "str_a": tri_blocksize * tri_blocksize,
                            "str_b": tri_blocksize * tri_blocksize,
                            "lda": tri_blocksize,
                            "ldb": tri_blocksize,
                            "ldc": tri_blocksize,
                            "count": s_pad // tri_blocksize,
                            "beta": 0.0,
                            "a_trans": 0,
                            "b_vnni": 1,
                            "transa": "N",
                            "transb": "N",
                        },
                    ),
                    (
                        "equation_brgemm_incoming",
                        {
                            "M": tri_blocksize,
                            "N": tri_blocksize,
                            "K": tri_blocksize,
                            "str_a": tri_blocksize * tri_blocksize,
                            "str_b": tri_blocksize * tri_blocksize,
                            "lda": tri_blocksize,
                            "ldb": tri_blocksize,
                            "ldc": tri_blocksize,
                            "count": b_pad // tri_blocksize,
                            "beta": 0.0,
                            "a_trans": 1,
                            "b_vnni": 1,
                            "transa": "T",
                            "transb": "N",
                        },
                    ),
                    (
                        "outgate_brgemm",
                        {
                            "M": tri_blocksize,
                            "N": act_dim,
                            "K": act_dim,
                            "lda": act_dim,
                            "ldb": act_dim,
                            "ldc": act_dim,
                            "count": 1,
                            "beta": 0.0,
                            "a_trans": 0,
                            "b_vnni": 1,
                            "transa": "N",
                            "transb": "N",
                        },
                    ),
                    (
                        "module_layout",
                        {
                            "B": b,
                            "S": s,
                            "B_pad": b_pad,
                            "S_pad": s_pad,
                            "act_dim": act_dim,
                            "num_intermediate_channel": num_intermediate_channel,
                            "tri_blocksize": tri_blocksize,
                        },
                    ),
                ],
            )
            print("[SME-GEMM-dev DEBUG]:TriangleMultiplication compare basis: smelt vs libxsmm (baseline is libxsmm)")
            with brgemm_backend(BrgemmBackend.LIBXSMM, verbose=False):
                ref_start = time.perf_counter()
                ref_act = _run_triangle_multiplication_impl(
                    input_act,
                    mask,
                    self.c_equation,
                    self.layer_norm_input.weight,
                    self.layer_norm_input.bias,
                    self.left_projection.weight,
                    self.left_projection.bias,
                    self.right_projection.weight,
                    self.right_projection.bias,
                    self.left_gate.weight,
                    self.left_gate.bias,
                    self.right_gate.weight,
                    self.right_gate.bias,
                    self.center_layer_norm.weight,
                    self.center_layer_norm.bias,
                    self.output_projection.weight,
                    self.output_projection.bias,
                    self.gating_linear.weight,
                    self.gating_linear.bias,
                )
                ref_elapsed = time.perf_counter() - ref_start
            log_brgemm_output_compare("TriangleMultiplication", smelt_act, ref_act)
            log_brgemm_timing_compare(
                "TriangleMultiplication", smelt_elapsed, ref_elapsed
            )
    return smelt_act


class TriangleMultiplicationOpti(nn.Module):

    #   def __init__(self,config, global_config, act_dim):
    def __init__(self, equation, num_intermediate_channel, act_dim):
        """Builds TriangleMultiplication module.

        Arguments:
          act: Pair activations, shape [N_res, N_res, c_z]
          mask: Pair mask, shape [N_res, N_res].
          is_training: Whether the module is in training mode.

        Returns:
          Outputs, same shape/type as act.
        """
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        # self.c_equation = self.config['equation']
        self.c_equation = equation
        # self.num_intermediate_channel = num_intermediate_channel
        self.layer_norm_input = nn.LayerNorm(
            normalized_shape=act_dim, elementwise_affine=True
        )
        self.left_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.right_projection = nn.Linear(act_dim, num_intermediate_channel)
        self.left_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.right_gate = nn.Linear(act_dim, num_intermediate_channel)
        self.center_layer_norm = nn.LayerNorm(
            normalized_shape=act_dim, elementwise_affine=True
        )
        self.output_projection = nn.Linear(act_dim, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask, backend=None):
        return TriangleMultiplicationOpti_forward(
            self, act, mask, backend=backend
        )
