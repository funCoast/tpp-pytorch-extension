/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#include "init.h"
#include "utils.h"
#include "xsmm_functors.h"

thread_local unsigned int* rng_state = NULL;
thread_local struct drand48_data drng_state; // For non AVX512 version

unsigned int saved_seed = 0;
void xsmm_manual_seed(unsigned int seed) {
  saved_seed = seed;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
#ifdef __x86_64__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

    if (rng_state) {
      libxsmm_rng_destroy_extstate(rng_state);
      rng_state = NULL;
    }
    rng_state = libxsmm_rng_create_extstate(seed + tid);
    srand48_r(seed + tid, &drng_state);
  }
}

unsigned int* get_rng_state() {
  if (rng_state) {
    return rng_state;
  }
  auto tid = omp_get_thread_num();
  rng_state = libxsmm_rng_create_extstate(saved_seed + tid);
  srand48_r(saved_seed + tid, &drng_state);
  return rng_state;
}

void init_libxsmm() {
  auto max_threads = omp_get_max_threads();
  TPP_ASSERT(
      max_threads <= MAX_THREADS,
      "Maximun %d threads supported, %d threads being used, please compile with increased  MAX_THREADS value\n",
      MAX_THREADS,
      max_threads);
  libxsmm_init();
  xsmm_manual_seed(0);
}

int get_brgemm_backend() {
  return static_cast<int>(tpp::get_brgemm_backend());
}

void set_brgemm_backend(int backend) {
  auto backend_enum = static_cast<tpp::BrgemmBackend>(backend);
  tpp::set_brgemm_backend(backend_enum);
#ifdef TPP_WITH_SMELT
  if (backend_enum == tpp::BrgemmBackend::SMELT) {
    SMELT::set_auto_context_switch(true);
  } else {
    SMELT::set_auto_context_switch(false);
  }
#endif
}

void set_smelt_auto_context_switch(bool enabled) {
#ifdef TPP_WITH_SMELT
  SMELT::set_auto_context_switch(enabled);
#else
  (void)enabled;
#endif
}

int get_vnni_blocking(py::object dtype) {
  c10::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
  return tpp::get_vnni_block_size(type);
}

REGISTER_SUBMODULE(_xsmm, m) {
  m.def("manual_seed", &xsmm_manual_seed, "Set libxsmm random seed");
  m.def("init_libxsmm", &init_libxsmm, "Initialize libxsmm");
  m.def("get_brgemm_backend", &get_brgemm_backend, "Get BrGEMM backend");
  m.def("set_brgemm_backend", &set_brgemm_backend, "Set BrGEMM backend");
  m.def(
      "set_smelt_auto_context_switch",
      &set_smelt_auto_context_switch,
      "Enable or disable SMELT auto context switching");
  m.def("get_vnni_blocking", &get_vnni_blocking, "Get VNNI pack size");
}
