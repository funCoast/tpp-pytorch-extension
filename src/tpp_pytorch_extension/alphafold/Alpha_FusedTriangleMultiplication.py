###############################################################################
# Copyright (c) 2023 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                    #
###############################################################################


import math
import time
import torch
from torch import nn
from torch.autograd import Function

from tpp_pytorch_extension._C import (
    _alpha_attention as Alpha_FusedTriangleMultiplication_cpp,
)
from tpp_pytorch_extension.utils.xsmm import (
    BrgemmBackend,
    brgemm_backend,
    is_smelt_backend,
    log_brgemm_output_compare,
    log_brgemm_shapes,
    log_brgemm_timing_compare,
    should_compare_brgemm,
)


class FusedTriangleMultiplicationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        act,
        mask,
        c_equation,
        left_norm_input_weight,
        left_norm_input_bias,
        projection_weight,
        projection_bias,
        gate_weight,
        gate_bias,
        center_norm_weight,
        center_norm_bias,
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
        act = Alpha_FusedTriangleMultiplication_cpp.fusedtrianglemulti_forward(
            act.contiguous(),
            mask.contiguous(),
            equation_flag,
            left_norm_input_weight,
            left_norm_input_bias,
            projection_weight,
            projection_bias,
            gate_weight,
            gate_bias,
            center_norm_weight,
            center_norm_bias,
            output_projection_weight,
            output_projection_bias,
            gating_linear_weight,
            gating_linear_bias,
        )
        return act


def _run_fused_triangle_multiplication_impl(
    act,
    mask,
    c_equation,
    left_norm_input_weight,
    left_norm_input_bias,
    projection_weight,
    projection_bias,
    gate_weight,
    gate_bias,
    center_norm_weight,
    center_norm_bias,
    output_projection_weight,
    output_projection_bias,
    gating_linear_weight,
    gating_linear_bias,
):
    if (
        act.dtype == torch.float16
        or mask.dtype == torch.float16
        or left_norm_input_weight.dtype == torch.float16
        or left_norm_input_bias.dtype == torch.float16
        or projection_weight.dtype == torch.float16
        or projection_bias.dtype == torch.float16
        or gate_weight.dtype == torch.float16
        or gate_bias.dtype == torch.float16
        or center_norm_weight.dtype == torch.float16
        or center_norm_bias.dtype == torch.float16
        or output_projection_weight.dtype == torch.float16
        or output_projection_bias.dtype == torch.float16
        or gating_linear_weight.dtype == torch.float16
        or gating_linear_bias.dtype == torch.float16
    ):
        return FusedTriangleMultiplicationFunction.apply(
            act.to(torch.float16),
            mask.to(torch.float32),
            c_equation,
            left_norm_input_weight.to(torch.float16),
            left_norm_input_bias.to(torch.float16),
            projection_weight.to(torch.float16),
            projection_bias.to(torch.float32),
            gate_weight.to(torch.float16),
            gate_bias.to(torch.float32),
            center_norm_weight.to(torch.float16),
            center_norm_bias.to(torch.float16),
            output_projection_weight.to(torch.float16),
            output_projection_bias.to(torch.float32),
            gating_linear_weight.to(torch.float16),
            gating_linear_bias.to(torch.float32),
        )

    return FusedTriangleMultiplicationFunction.apply(
        act,
        mask,
        c_equation,
        left_norm_input_weight,
        left_norm_input_bias,
        projection_weight,
        projection_bias,
        gate_weight,
        gate_bias,
        center_norm_weight,
        center_norm_bias,
        output_projection_weight,
        output_projection_bias,
        gating_linear_weight,
        gating_linear_bias,
    )


def FusedTriangleMultiplicationOpti_forward(self, act, mask, backend=None):
    mask = mask[..., None]
    with brgemm_backend(backend):
        input_act = act
        compare_active = should_compare_brgemm() and is_smelt_backend()
        if compare_active:
            compare_start = time.perf_counter()
        smelt_act = _run_fused_triangle_multiplication_impl(
            input_act,
            mask,
            self.c_equation,
            self.left_norm_input.weight,
            self.left_norm_input.bias,
            self.projection.weight,
            self.projection.bias,
            self.gate.weight,
            self.gate.bias,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection.weight,
            self.output_projection.bias,
            self.gating_linear.weight,
            self.gating_linear.bias,
        )
        if compare_active:
            smelt_elapsed = time.perf_counter() - compare_start

        if compare_active:
            log_brgemm_shapes(
                "FusedTriangleMultiplication",
                [
                    ("act", input_act),
                    ("mask", mask),
                    ("left_norm_input.weight", self.left_norm_input.weight),
                    ("left_norm_input.bias", self.left_norm_input.bias),
                    ("projection.weight", self.projection.weight),
                    ("projection.bias", self.projection.bias),
                    ("gate.weight", self.gate.weight),
                    ("gate.bias", self.gate.bias),
                    ("center_norm.weight", self.center_norm.weight),
                    ("center_norm.bias", self.center_norm.bias),
                    ("output_projection.weight", self.output_projection.weight),
                    ("output_projection.bias", self.output_projection.bias),
                    ("gating_linear.weight", self.gating_linear.weight),
                    ("gating_linear.bias", self.gating_linear.bias),
                ],
            )
            with brgemm_backend(BrgemmBackend.LIBXSMM):
                ref_start = time.perf_counter()
                ref_act = _run_fused_triangle_multiplication_impl(
                    input_act,
                    mask,
                    self.c_equation,
                    self.left_norm_input.weight,
                    self.left_norm_input.bias,
                    self.projection.weight,
                    self.projection.bias,
                    self.gate.weight,
                    self.gate.bias,
                    self.center_norm.weight,
                    self.center_norm.bias,
                    self.output_projection.weight,
                    self.output_projection.bias,
                    self.gating_linear.weight,
                    self.gating_linear.bias,
                )
                ref_elapsed = time.perf_counter() - ref_start
            log_brgemm_output_compare(
                "FusedTriangleMultiplication", smelt_act, ref_act
            )
            log_brgemm_timing_compare(
                "FusedTriangleMultiplication", smelt_elapsed, ref_elapsed
            )
    return smelt_act


class FusedTriangleMultiplicationOpti(nn.Module):

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
        self.left_norm_input = nn.LayerNorm(act_dim)
        self.projection = nn.Linear(act_dim, 2 * num_intermediate_channel)
        self.gate = nn.Linear(num_intermediate_channel, 2 * num_intermediate_channel)
        self.center_norm = nn.LayerNorm(num_intermediate_channel)
        self.output_projection = nn.Linear(num_intermediate_channel, act_dim)
        self.gating_linear = nn.Linear(act_dim, act_dim)

    def forward(self, act, mask, backend=None):
        return FusedTriangleMultiplicationOpti_forward(
            self, act, mask, backend=backend
        )
