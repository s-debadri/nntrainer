// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.hpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#ifndef GPU_CL_OPENCL_KERNEL_HPP_
#define GPU_CL_OPENCL_KERNEL_HPP_

#include <string>

#include "opencl_program.hpp"
#include "third_party/cl.h"

namespace nntrainer::internal {
class Kernel {
  cl_kernel kernel_{nullptr};

public:
  bool CreateKernelFromProgram(Program program,
                               const std::string &function_name);

  bool SetKernelArguments(cl_uint arg_index, const void *arg_value,
                          size_t size);
  const cl_kernel GetKernel();
};
} // namespace nntrainer::internal
#endif // GPU_CL_OPENCL_KERNEL_HPP_