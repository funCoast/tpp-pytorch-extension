###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Narendra Chaudhary (Intel Corp.)                                       #
###############################################################################

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from tpp_pytorch_extension.utils.xsmm import (
    BrgemmBackend,
    brgemm_backend,
    get_brgemm_backend,
    is_smelt_backend,
    log_brgemm_output_compare,
    log_brgemm_shapes,
    log_brgemm_timing_compare,
    should_compare_brgemm,
)

# from tpp_pytorch_extension.utils.blocked_layout import (
#     BlockedParameter,
#     BlockedModule,
#     BlockedTensor,
#     get_blocking_signature,
# )
from tpp_pytorch_extension._C import _alpha_attention as Alpha_Attention_cpp
import time
class AlphaAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        q_data,
        m_data,
        bias,
        nonbatched_bias,
        query_w,
        key_w,
        value_w,
        gating_w,
        gating_b,
        output_w,
        output_b,
        key_dim,
        value_dim,
    ):
        result = Alpha_Attention_cpp.forward(
            q_data,
            m_data,
            bias,
            nonbatched_bias,
            query_w,
            key_w,
            value_w,
            gating_w,
            gating_b,
            output_w,
            output_b,
            key_dim,
            value_dim,
        )
        return result


def _run_gating_attention_impl(
    q_data,
    m_data,
    bias,
    nonbatched_bias,
    query_w,
    key_w,
    value_w,
    gating_w,
    gating_b,
    output_w,
    output_b,
    key_dim,
    value_dim,
):
    if (
        q_data.dtype == torch.float16
        or m_data.dtype == torch.float16
        or bias.dtype == torch.float16
        or nonbatched_bias.dtype == torch.float16
        or query_w.dtype == torch.float16
        or key_w.dtype == torch.float16
        or value_w.dtype == torch.float16
        or gating_w.dtype == torch.float16
        or gating_b.dtype == torch.float16
        or output_w.dtype == torch.float16
        or output_b.dtype == torch.float16
    ):
        return AlphaAttentionFunction.apply(
            q_data.type(torch.float16),
            m_data.type(torch.float16),
            bias.type(torch.float32),
            nonbatched_bias.type(torch.float32),
            query_w.type(torch.float16),
            key_w.type(torch.float16),
            value_w.type(torch.float16),
            gating_w.type(torch.float16),
            gating_b.type(torch.float32),
            output_w.type(torch.float16),
            output_b.type(torch.float32),
            key_dim,
            value_dim,
        )

    return AlphaAttentionFunction.apply(
        q_data,
        m_data,
        bias,
        nonbatched_bias,
        query_w,
        key_w,
        value_w,
        gating_w,
        gating_b,
        output_w,
        output_b,
        key_dim,
        value_dim,
    )


def GatingAttentionOpti_forward(
    self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(), backend=None
):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """

    # output = AlphaAttentionFunction.apply(
    #     q_data,
    #     m_data,
    #     bias,
    #     nonbatched_bias,
    #     self.query_w,
    #     self.key_w,
    #     self.value_w,
    #     self.gating_w,
    #     self.gating_b,
    #     self.output_w,
    #     self.output_b,
    #     self.key_dim,
    #     self.value_dim,
    # )

    effective_backend = get_brgemm_backend() if backend is None else backend
    if (
        torch.is_tensor(q_data)
        and q_data.ndim > 0
        and q_data.shape[0] == 320
        and is_smelt_backend(effective_backend)
    ):
        effective_backend = BrgemmBackend.LIBXSMM

    with brgemm_backend(effective_backend):
        smelt_elapsed = None
        ref_elapsed = None
        compare_active = should_compare_brgemm() and is_smelt_backend()
        if compare_active:
            compare_start = time.perf_counter()
        output = _run_gating_attention_impl(
            q_data,
            m_data,
            bias,
            nonbatched_bias,
            self.query_w,
            self.key_w,
            self.value_w,
            self.gating_w,
            self.gating_b,
            self.output_w,
            self.output_b,
            self.key_dim,
            self.value_dim,
        )
        if compare_active:
            smelt_elapsed = time.perf_counter() - compare_start

        if compare_active:
            log_brgemm_shapes(
                "AlphaAttention",
                [
                    ("q_data", q_data),
                    ("m_data", m_data),
                    ("bias", bias),
                    ("nonbatched_bias", nonbatched_bias),
                    ("query_w", self.query_w),
                    ("key_w", self.key_w),
                    ("value_w", self.value_w),
                    ("gating_w", self.gating_w),
                    ("gating_b", self.gating_b),
                    ("output_w", self.output_w),
                    ("output_b", self.output_b),
                ],
            )
            with brgemm_backend(BrgemmBackend.LIBXSMM):
                ref_start = time.perf_counter()
                ref_output = _run_gating_attention_impl(
                    q_data,
                    m_data,
                    bias,
                    nonbatched_bias,
                    self.query_w,
                    self.key_w,
                    self.value_w,
                    self.gating_w,
                    self.gating_b,
                    self.output_w,
                    self.output_b,
                    self.key_dim,
                    self.value_dim,
                )
                ref_elapsed = time.perf_counter() - ref_start
            log_brgemm_output_compare("AlphaAttention", output, ref_output)
            log_brgemm_timing_compare(
                "AlphaAttention", smelt_elapsed, ref_elapsed
            )
    return output


class GatingAttentionOpti(nn.Module):
    """Multihead attention w/ Gating"""

    # def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    def __init__(self, num_head, a_dim, m_dim, output_dim):
        super().__init__()
        # self.config = config
        # self.global_config = global_config
        self.output_dim = output_dim
        # k,v dim
        # self.key_dim = self.config.get('key_dim', int(a_dim))
        # self.value_dim = self.config.get('value_dim', int(m_dim))
        # self.num_head = self.config['num_head']
        self.key_dim = int(a_dim)
        self.value_dim = int(m_dim)
        self.num_head = num_head
        assert self.key_dim % self.num_head == 0
        assert self.value_dim % self.num_head == 0
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        # q,k,v weights
        self.query_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.key_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.key_dim), requires_grad=False
        )
        self.value_w = nn.Parameter(
            torch.Tensor(m_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_w = nn.Parameter(
            torch.Tensor(a_dim, self.num_head, self.value_dim), requires_grad=False
        )
        self.gating_b = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim), requires_grad=False
        )
        self.output_w = nn.Parameter(
            torch.Tensor(self.num_head, self.value_dim, self.output_dim),
            requires_grad=False,
        )
        self.output_b = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=False)
        # softmax & act fn
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    @torch.jit.ignore
    def read_time(self) -> float:
        return time.time()

    def forward(
        self, q_data, m_data, bias, nonbatched_bias=torch.Tensor(), backend=None
    ):
        """Builds Attention module.
        Arguments:
          q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
          m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
          bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
          nonbatched_bias: Shared bias, shape [N_queries, N_keys].
        Returns:
          A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        # inputs = [q_data, m_data, bias, nonbatched_bias]
        # inputs += [self.query_w, self.key_w, self.value_w]
        # inputs += [self.gating_w, self.gating_b]
        # inputs += [self.output_w, self.output_b]

        # layer_dtype = q_data.dtype
        # inputs = [i.to(layer_dtype) if i.is_floating_point() else i for i in inputs]

        # output = AlphaAttentionFunction.apply(*inputs, self.key_dim, self.value_dim)

        return GatingAttentionOpti_forward(
            self,
            q_data,
            m_data,
            bias,
            nonbatched_bias=nonbatched_bias,
            backend=backend,
        )
