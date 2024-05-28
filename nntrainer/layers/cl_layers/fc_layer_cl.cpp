/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include "fc_layer_cl.h"
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <blas_interface.h>

#include <iostream>

#define sgemv_loop(ci, cj, cM, cN)                         \
  do {                                                     \
    float y0;                                              \
    unsigned int i, j;                                     \
    for (ci = 0; ci != cM; ci++) {                         \
      y0 = vecYdata[ci * incy] * 0.0f;                     \
      for (cj = 0; cj != cN; cj++)                         \
        y0 += matAdata[i + j * lda] * vecXdata[cj * incx]; \
      vecYdata[ci * incy] = y0;                            \
    }                                                      \
  } while (0);

std::string fc_sgemv_kernel_ =
  R"(__kernel void sgemv(const __global float* A, const __global float* X,
                      __global float* Y, unsigned int M, unsigned int N) {
        // const int row = get_global_id(0);
        // for (unsigned int j = 0; j < N; j++){
        //     Y[row] +=  A[row * N + j] * X[j];
        // }
        float y0;                                              
        unsigned int i, j;                                     
        for (unsigned int i = 0; i < N; i++) {                        
            float y0 = Y[i] * 0.0f;
            for (unsigned int j = 0; j < M; j++)                         
                y0 += A[i + j * N] * X[j]; 
            Y[i] = y0;                            
        }     
    })";

std::string fc_dot_kernel_ =
  R"(__kernel void dot(const __global float* A, const __global float* X, unsigned int K, float res) {
        res = 0;
        for (unsigned int i = 0; i < K; i++){
            res += A[i] * X[i];`
        }
    })";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

FullyConnectedLayerCl::FullyConnectedLayerCl() :
  LayerImpl(), fc_props(props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedLayerCl::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }
}

void FullyConnectedLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayerCl::forwarding(RunLayerContext &context, bool training) {
  // #ifdef ENABLE_OPENCL
  //   ml_logi("FC Layer [forwarding]: compute engine -> %d",
  //           context.getComputeEngine());
  // #endif
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (weight.getDataType() == nntrainer::Tdatatype::QINT4 ||
      weight.getDataType() == nntrainer::Tdatatype::QINT8) {
    Tdatatype dtype = input_.getDataType();

    Tensor weight_(
      {{weight.batch(), weight.channel(), weight.height(), weight.width()},
       {weight.getFormat(), dtype}},
      true);

    unsigned int axis =
      context.getWeightObject(weight_idx[FCParams::weight]).getOutputAxis();

    weight.dequantize(weight_, axis);
#ifdef ENABLE_OPENCL
    clForward(input_, weight_, hidden_, context);
#else
    input_.dot(weight_, hidden_, false, false);
#endif
  } else {
#ifdef ENABLE_OPENCL
    clForward(input_, weight, hidden_, context);
#else
    input_.dot(weight, hidden_, false, false);
#endif
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

#ifdef ENABLE_OPENCL

/**
 * @brief declaring static kernel obj
 *
 */
opencl::Kernel FullyConnectedLayerCl::kernel_;

void FullyConnectedLayerCl::clForward(Tensor const &input, Tensor const &weight,
                                    Tensor &result, RunLayerContext &context) {
  // to do:
  // NNTR_THROW_IF(!contiguous, std::invalid_argument)
  //   << getName() << " is not contiguous. Cannot dot product.";

  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = weight.batch() * weight.height() * weight.width();
    mdim2 = weight.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = weight.batch() * weight.channel() * weight.height();
    mdim2 = weight.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;
  if (dim2 != mdim1)
    throw std::runtime_error("Error: incompatible dimensions for dot product");
  K = mdim1; /** == dim2 */
  N = mdim2;
  M = dim1;
  if (input.getFormat() == Tformat::NHWC) {
    CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                         input.width(),
                         input.getTensorType()); //  NHWC Result Tensor
  } else {
    CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                         N, input.getTensorType());
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = weight.getData();
    float *rdata = result.getData();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    enum CBLAS_TRANSPOSE transA = CblasNoTrans;
    enum CBLAS_TRANSPOSE transB = CblasNoTrans;

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = clDot(data, mdata, K, context) + beta * (*rdata);
      // *rdata = cl_kernel_sdot(K, data, 1, mdata, 1) + beta * (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      ml_logi("fc_layer: clForward N==1");
      sgemv(true, data, mdata, rdata, dim1, dim2, lda, context);

      // ::nntrainer::sgemv(CblasRowMajor, transA, dim1, dim2, alpha, data, lda,
      // mdata, 1, beta,
      //       rdata, 1);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      ml_logi("fc_layer: clForward M==1");
      sgemv(true, mdata, data, rdata, mdim1, mdim2, ldb, context);

      // transB = transB == CblasTrans ? CblasNoTrans : CblasTrans;
      // ::nntrainer::sgemv(CblasRowMajor, transB, mdim1, mdim2, alpha, mdata,
      // ldb, data, 1,
      //       beta, rdata, 1);
    }
    /// case others: use gemm
    else {
      throw std::invalid_argument(
        "Error: OpenCL fc_layer sgemm not implemented yet.");
      // sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, data, lda, mdata,
      //       ldb, beta, rdata, ldc);
    }
  } else
    throw std::invalid_argument("Error: OpenCL fp16 is not supported yet.");
}

void FullyConnectedLayerCl::sgemv(bool transA, const float *matAdata,
                                const float *vecXdata, float *vecYdata,
                                unsigned int dim1, unsigned int dim2,
                                unsigned int lda, RunLayerContext &context) {
  ml_logi("fc_layer::sgemv");
  // unsigned int incx = 1, incy = 1;
  // sgemv_loop(i, j, dim2, dim1);

  bool result = false;

  do {
    result = context.clCreateKernel(fc_sgemv_kernel_, context.LayerKernel::SGEMV,
                                    FullyConnectedLayerCl::kernel_);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;
    size_t dim2_size = sizeof(float) * dim2;
    opencl::Buffer inputA(context.context_inst_, dim1_size * dim2_size, true,
                          nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inOutY(context.context_inst_, dim2_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = inOutY.WriteData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(0, &inputA,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(1, &inputX,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(2, &inOutY,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      FullyConnectedLayerCl::kernel_.SetKernelArguments(3, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result =
      FullyConnectedLayerCl::kernel_.SetKernelArguments(4, &dim2, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      FullyConnectedLayerCl::kernel_, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = inOutY.ReadData(context.command_queue_inst_, vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

float FullyConnectedLayerCl::clDot(const float *matAdata, const float *vecXdata,
                                 unsigned int dim1, RunLayerContext &context) {
  ml_logi("fc_layer::clDot");

  bool result = false;

  float cl_ret = 0;

  do {
    // FullyConnectedLayerCl::kernel_ is wrong for this ...its sgemv.
    result = context.clCreateKernel(fc_dot_kernel_, context.LayerKernel::DOT,
                                    FullyConnectedLayerCl::kernel_);
    if (!result) {
      break;
    }

    size_t dim1_size = sizeof(float) * dim1;

    opencl::Buffer inputA(context.context_inst_, dim1_size, true, nullptr);

    opencl::Buffer inputX(context.context_inst_, dim1_size, true, nullptr);

    result = inputA.WriteData(context.command_queue_inst_, matAdata);
    if (!result) {
      break;
    }

    result = inputX.WriteData(context.command_queue_inst_, vecXdata);
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(0, &inputA,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(1, &inputX,
                                                             sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      FullyConnectedLayerCl::kernel_.SetKernelArguments(2, &dim1, sizeof(int));
    if (!result) {
      break;
    }

    result = FullyConnectedLayerCl::kernel_.SetKernelArguments(3, &cl_ret,
                                                             sizeof(float));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    result = context.command_queue_inst_.DispatchCommand(
      FullyConnectedLayerCl::kernel_, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

  } while (false);

  return cl_ret;
}
#endif

void FullyConnectedLayerCl::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor w;
  Tensor &weight = w;
  context.getWeight(weight, weight_idx[FCParams::weight]);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  input_step_dim.height(to - from);
  hidden_step_dim.height(to - from);

  // @todo: set reset stride as false. This implementation only works when batch
  // size is 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  input_step.dot(weight, hidden_step, false, false);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_step.add_i(bias);
  }
}

void FullyConnectedLayerCl::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void FullyConnectedLayerCl::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);

    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      /// @todo optimize below by adding beta to Tensor::sum
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} /* namespace nntrainer */
