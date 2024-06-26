// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_v2.cpp
 * @date	01 December 2023
 * @brief	This is a TensorV2 class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <float_tensor.h>
#include <tensor_v2.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif

namespace nntrainer {

TensorV2::TensorV2(std::string name_, Tformat fm, Tdatatype d_type) {
  itensor = nullptr;

  if (d_type == Tdatatype::FP32) {
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(name_, fm),
                                           std::default_delete<FloatTensor>());
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(name_, fm),
                                          std::default_delete<HalfTensor>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(const TensorDim &d, bool alloc_now, Initializer init,
                   std::string name) {
  itensor = nullptr;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor =
      std::shared_ptr<FloatTensor>(new FloatTensor(d, alloc_now, init, name),
                                   std::default_delete<FloatTensor>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor =
      std::shared_ptr<HalfTensor>(new HalfTensor(d, alloc_now, init, name),
                                  std::default_delete<HalfTensor>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(const TensorDim &d, const void *buf) {
  itensor = nullptr;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(d, buf),
                                           std::default_delete<FloatTensor>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(d, buf),
                                          std::default_delete<HalfTensor>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor = std::shared_ptr<FloatTensor>(new FloatTensor(d, t_type.format),
                                         std::default_delete<FloatTensor>());
}

#ifdef ENABLE_FP16
TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor = std::shared_ptr<HalfTensor>(new HalfTensor(d, t_type.format),
                                        std::default_delete<HalfTensor>());
}
#endif

bool TensorV2::operator==(const TensorV2 &rhs) const {
  /// compares tensor information
  if (*itensor == *rhs.itensor) {
    /// compares tensor data
    if (getDataType() == Tdatatype::FP32) {
      return *std::dynamic_pointer_cast<FloatTensor>(itensor) ==
             *std::dynamic_pointer_cast<FloatTensor>(rhs.itensor);
    } else if (getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      return *std::dynamic_pointer_cast<HalfTensor>(itensor) ==
             *std::dynamic_pointer_cast<HalfTensor>(rhs.itensor);
#else
      throw std::invalid_argument(
        "Error: HalfTensor cannot be created or used when FP16 is not enabled. "
        "Please check if the tensor data type is set properly.");
#endif
    }
  }
  return false;
}

void TensorV2::allocate() { itensor->allocate(); }

void TensorV2::deallocate() { itensor->deallocate(); }

bool TensorV2::isAllocated() { return itensor->isAllocated(); }

void TensorV2::setValue(float value) { itensor->setValue(value); }

void TensorV2::setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) {
  itensor->setValue(b, c, h, w, value);
}

void TensorV2::addValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value, float beta) noexcept {
  itensor->addValue(b, c, h, w, value, beta);
}

void TensorV2::setZero() { itensor->setZero(); }

void TensorV2::setRandNormal(float mean, float stddev) {
  itensor->setRandNormal(mean, stddev);
}

void TensorV2::setRandUniform(float min, float max) {
  itensor->setRandUniform(min, max);
}

void TensorV2::setRandBernoulli(float probability) {
  itensor->setRandBernoulli(probability);
}

void TensorV2::initialize() { itensor->initialize(); }

void TensorV2::initialize(Initializer init) { itensor->initialize(init); }

TensorV2 TensorV2::apply(std::function<TensorV2(TensorV2)> f) const {
  return f(*this);
}

TensorV2 &TensorV2::apply(std::function<TensorV2 &(TensorV2, TensorV2 &)> f,
                          TensorV2 &output) const {
  return f(*this, output);
}

int TensorV2::multiply_i_strided(TensorV2 const &m, const float beta) {
  try {
    this->multiply_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::multiply_strided(TensorV2 const &m, const float beta) const {
  TensorV2 t;
  return this->multiply_strided(m, t, beta);
}

TensorV2 &TensorV2::multiply_strided(TensorV2 const &m, TensorV2 &output,
                                     const float beta) const {
  itensor->multiply_strided(m, output, beta);
  return output;
}

int TensorV2::multiply_i(float const &value) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  return itensor->multiply_i(value);
}

TensorV2 TensorV2::multiply(float const &value) const {
  TensorV2 t;
  return multiply(value, t);
}

TensorV2 &TensorV2::multiply(float const &value, TensorV2 &out) const {
  itensor->multiply(value, out);
  return out;
}

int TensorV2::multiply_i(TensorV2 const &m, const float beta) {
  try {
    this->multiply(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::multiply(TensorV2 const &m, const float beta) const {
  TensorV2 t("", this->getFormat());
  return multiply(m, t, beta);
}

TensorV2 &TensorV2::multiply(TensorV2 const &m, TensorV2 &output,
                             const float beta) const {
  itensor->multiply(m, output, beta);
  return output;
}

int TensorV2::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  this->divide(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::divide(float const &value) const {
  TensorV2 output("", getFormat(), getDataType());
  return divide(value, output);
}

TensorV2 &TensorV2::divide(float const &value, TensorV2 &output) const {
  /// @todo add unittest, ZeroDivisionError
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[Tensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }
  itensor->divide(value, output);
  return output;
}

int TensorV2::divide_i(TensorV2 const &m) {
  try {
    this->divide(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::divide(TensorV2 const &m) const {
  TensorV2 output("", getFormat(), getDataType());
  return this->divide(m, output);
}

TensorV2 &TensorV2::divide(TensorV2 const &m, TensorV2 &output) const {
  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot divide";
  itensor->divide(m, output);
  return output;
}

int TensorV2::add_i_strided(TensorV2 const &input, const float beta) {
  try {
    this->add_strided(input, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add_strided(TensorV2 const &input, const float beta) const {
  TensorV2 output("", getFormat(), getDataType());
  return this->add_strided(input, output, beta);
}

TensorV2 &TensorV2::add_strided(TensorV2 const &input, TensorV2 &output,
                                const float beta) const {
  CREATE_V2_IF_EMPTY_DIMS(output, getDim(), nullptr);

  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  itensor->add_strided(input, output, beta);

  return output;
}

int TensorV2::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add(float const &value) const {
  TensorV2 t("", getFormat(), getDataType());
  return add(value, t);
}

TensorV2 &TensorV2::add(float const &value, TensorV2 &output) const {
  itensor->add(value, output);
  return output;
}

int TensorV2::add_i(TensorV2 const &m, float const alpha) {
  try {
    this->add(m, *this, alpha);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add(TensorV2 const &m, float const alpha) const {
  TensorV2 t("", getFormat(), getDataType());
  return this->add(m, t, alpha);
}

TensorV2 &TensorV2::add(TensorV2 const &m, TensorV2 &output,
                        float const alpha) const {
  NNTR_THROW_IF(!itensor->getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";
  itensor->add(m, output, alpha);
  return output;
}

int TensorV2::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::subtract(float const &value) const {
  TensorV2 output("", getFormat(), getDataType());
  return subtract(value, output);
}

TensorV2 &TensorV2::subtract(float const &value, TensorV2 &output) const {
  itensor->subtract(value, output);
  return output;
}

int TensorV2::subtract_i(TensorV2 const &m) { return add_i(m, -1); }

TensorV2 TensorV2::subtract(TensorV2 const &m) const { return add(m, -1); }

TensorV2 &TensorV2::subtract(TensorV2 const &m, TensorV2 &output) const {
  return add(m, output, -1);
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
TensorV2 TensorV2::sum_by_batch() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  TensorV2 output(batch(), 1, 1, 1, this->getFormat(), getDataType());
  itensor->sum_by_batch(output);
  return output;
}

TensorV2 TensorV2::sum(unsigned int axis, float alpha) const {
  TensorV2 output("", this->getFormat(), this->getDataType());
  return sum(axis, output, alpha, 0);
}

TensorV2 &TensorV2::sum(unsigned int axis, TensorV2 &output, float alpha,
                        float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  itensor->sum(axis, output, alpha, beta);
  return output;
}

TensorV2 TensorV2::sum(const std::vector<unsigned int> &axes,
                       float alpha) const {
  TensorV2 output("", this->getFormat());
  return sum(axes, output, alpha);
}

TensorV2 &TensorV2::sum(const std::vector<unsigned int> &axes, TensorV2 &output,
                        float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {

    /** club axes together */
    TensorV2 new_reshaped = TensorV2(getDim());
    new_reshaped.copy(*this);
    std::vector<unsigned int> continuous_order = {0, 3, 1, 2};
    std::vector<unsigned int> new_axes = {axes[0]};

    for (unsigned int i = 1; i < axes.size(); ++i) {
      if (checkContinuous(axes[i - 1], axes[i])) {
        new_reshaped.mergeAxis(axes[i - 1], axes[i]);
        new_axes.back() = axes[i];
      } else {
        new_axes.push_back(axes[i]);
      }
    }

    TensorV2 ret = new_reshaped.sum(new_axes[0]);
    for (unsigned int i = 1; i < new_axes.size() - 1; ++i)
      ret = ret.sum(axes[i]);
    ret.sum(new_axes.back(), output, alpha);
  }
  return output;
}

TensorV2 TensorV2::average(unsigned int axis) const {
  TensorV2 output("", this->getFormat(), this->getDataType());
  return average(axis, output);
}

TensorV2 &TensorV2::average(unsigned int axis, TensorV2 &output) const {
  if (axis >= TensorDim::MAXDIM)
    throw std::out_of_range(
      "negative axis or axis more then MAXDIM is invalid");

  unsigned int axis_size = getDim()[axis];
  if (axis_size == 1)
    output.copy(*this);
  else
    this->sum(axis, output, 1.0 / ((float)axis_size));

  return output;
}

TensorV2 TensorV2::average(const std::vector<unsigned int> &axes) const {
  TensorV2 output("", this->getFormat(), this->getDataType());
  return average(axes, output);
}

TensorV2 &TensorV2::average(const std::vector<unsigned int> &axes,
                            TensorV2 &output) const {
  if (axes.empty())
    return this->average(output);

  TensorDim ret_shape(getTensorType());

  for (const auto &idx : axes) {
    if (idx >= TensorDim::MAXDIM) {
      throw std::out_of_range("axis more then MAXDIM is invalid");
    }
    ret_shape.setTensorDim(idx, getDim().getTensorDim(idx));
  }

  return this->sum(axes, output, 1.0 / (float)ret_shape.getDataLen());
}

TensorV2 TensorV2::average() const {
  TensorV2 output = *this;
  unsigned int axis = 0;
  if (this->getFormat() == Tformat::NHWC) {
    output.reshape({1, getDim().getDataLen(), 1, 1, this->getTensorType()});
    axis = 1;
  } else {
    output.reshape({1, 1, 1, getDim().getDataLen(), this->getTensorType()});
    axis = 3;
  }
  return output.average(axis);
}

TensorV2 &TensorV2::average(TensorV2 &output) const {
  TensorV2 result = *this;
  result.reshape({1, 1, 1, getDim().getDataLen()});
  return result.average(3, output);
}

int TensorV2::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::pow(float exponent) const {
  TensorV2 output("", getFormat(), getDataType());
  return pow(exponent, output);
}

TensorV2 &TensorV2::pow(float exponent, TensorV2 &output) const {
  itensor->pow(exponent, output);
  return output;
}

int TensorV2::erf_i() {
  erf(*this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::erf() const {
  TensorV2 output("", getFormat(), getDataType());
  return erf(output);
}

TensorV2 &TensorV2::erf(TensorV2 &output) const {
  itensor->erf(output);
  return output;
}

void TensorV2::sin(TensorV2 &out, float alpha) {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::sin must match");

  itensor->sin(out, alpha);
}

void TensorV2::cos(TensorV2 &out, float alpha) {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::cos must match");

  itensor->cos(out, alpha);
}

float TensorV2::l2norm() const { return itensor->l2norm(); }

void TensorV2::normalization_i() {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot do normalization.";

  const float min = minValue();
  const float max = maxValue();

  if (max == min) {
    TensorV2 tmp = *this;
    this->subtract_i(tmp);
  } else {
    this->subtract_i(min);
    this->divide_i(max - min);
  }
}

void TensorV2::standardization_i() {
  TensorV2 mean_by_batch = this->sum_by_batch();
  mean_by_batch.divide_i(getDim().getFeatureLen());

  this->subtract_i(mean_by_batch);
  TensorV2 std_dev_by_batch(batch(), 1, 1, 1, getFormat(), getDataType());
  std_dev_by_batch.setZero();

  /// @todo remove conditional statement
  if (getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *std_dev = std_dev_by_batch.getData<float>();

    for (unsigned int k = 0; k < batch(); ++k) {
      TensorV2 sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = sub_this.l2norm();
    }
  } else if (getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *std_dev = std_dev_by_batch.getData<_FP16>();

    for (unsigned int k = 0; k < batch(); ++k) {
      TensorV2 sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = static_cast<_FP16>(sub_this.l2norm());
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  std_dev_by_batch.divide_i(getDim().getFeatureLen());
  this->divide_i(std_dev_by_batch);
}

TensorV2 TensorV2::dot(TensorV2 const &input, bool trans, bool trans_in) const {
  TensorV2 output("", this->getFormat(), this->getDataType());
  dot(input, output, trans, trans_in);

  return output;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
TensorV2 &TensorV2::dot(TensorV2 const &input, TensorV2 &output, bool trans,
                        bool trans_in, float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  itensor->dot(input, output, trans, trans_in, beta);
  return output;
}

TensorV2 &TensorV2::dot_deriv_wrt_1(TensorV2 const &m,
                                    TensorV2 const &output_deriv, bool trans,
                                    bool trans_m, float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dot(m, *this, deriv_trans, deriv_trans_m, beta);
}

/**
 * @brief compute the derivative wrt m in the m tensor
 * @note The caller tensor must be the same tensor as the one which called the
 * dot() product.
 */
TensorV2 &TensorV2::dot_deriv_wrt_2(TensorV2 &m_deriv,
                                    TensorV2 const &output_deriv, bool trans,
                                    bool trans_m, float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dot(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dot(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

TensorV2 &TensorV2::dotBatched(TensorV2 const &m, TensorV2 &result, bool trans,
                               bool trans_m, float beta) const {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const TensorV2 this_b = this->getBatchSlice(b, 1);
    TensorV2 m_b = m.getBatchSlice(b, 1);
    TensorV2 result_b = result.getBatchSlice(b, 1);

    this_b.dot(m_b, result_b, trans, trans_m, beta);
  }

  return result;
}

TensorV2 &TensorV2::dot_batched_deriv_wrt_1(TensorV2 const &m,
                                            TensorV2 const &output_deriv,
                                            bool trans, bool trans_m,
                                            float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dotBatched(m, *this, deriv_trans, deriv_trans_m, beta);
}

TensorV2 &TensorV2::dot_batched_deriv_wrt_2(TensorV2 &m_deriv,
                                            TensorV2 const &output_deriv,
                                            bool trans, bool trans_m,
                                            float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dotBatched(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dotBatched(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

TensorV2 TensorV2::dropout_mask(float dropout) const {
  TensorV2 output(getDim());
  output.dropout_mask(dropout);
  return output;
}

void TensorV2::dropout_mask(float dropout) {
  /// @todo add unittest
  NNTR_THROW_IF(dropout < 0 || dropout > 1, std::invalid_argument)
    << "[Tensor::dropout_mask] Dropout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(dropout) == FP_ZERO)
    return;

  setRandUniform(0.0, 1.0);
  itensor->dropout_mask(dropout);
}

void TensorV2::filter_mask(const TensorV2 &mask_len, bool reverse) {
  /// @todo add unittest
  itensor->filter_mask(mask_len, reverse);
}

TensorV2 TensorV2::zoneout_mask(float zoneout) {
  TensorV2 output(getDim());
  zoneout_mask(output, zoneout);
  return output;
}

void TensorV2::zoneout_mask(TensorV2 &opposite, float zoneout) {
  NNTR_THROW_IF(getDim() != opposite.getDim(), std::invalid_argument)
    << "[Tensor::zoneout_mask] opposite dimension does not match";

  NNTR_THROW_IF(zoneout < 0 || zoneout > 1, std::invalid_argument)
    << "[Tensor::zoneout_mask] Zoneout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(zoneout) == FP_ZERO)
    return;

  itensor->zoneout_mask(opposite, zoneout);
}

std::vector<TensorV2> TensorV2::split(unsigned num_size, int axis) {
  NNTR_THROW_IF(num_size == 0, std::invalid_argument)
    << "num size cannot be zero";

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(getDim().getTensorDim(axis) % num_size != 0,
                std::invalid_argument)
    << "axis is not divisible by num_size, axis: " << axis
    << " num size: " << num_size;

  std::vector<size_t> sizes;
  sizes.resize(num_size);

  unsigned int sz = getDim().getTensorDim(axis) / num_size;
  std::fill(sizes.begin(), sizes.end(), sz);

  return split(sizes, axis);
}

std::vector<TensorV2> TensorV2::split(std::vector<size_t> sizes, int axis) {
  NNTR_THROW_IF(sizes.size() == 0, std::invalid_argument)
    << "num size cannot be zero";

  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(
    std::any_of(sizes.begin(), sizes.end(), [](size_t sz) { return !sz; }),
    std::invalid_argument)
    << "among given sizes at least one of size is 0";

  return itensor->split(sizes, axis);
}

TensorV2 TensorV2::cat(const std::vector<TensorV2> &tensors, int axis) {
  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(tensors.empty(), std::invalid_argument)
    << "given tensor vector is empty";

  TensorV2 output;
  Tdatatype dtype = tensors.front().getDim().getDataType();

  if (dtype == Tdatatype::FP32) {
    output = FloatTensor::cat(tensors, axis);
  } else if (dtype == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    output = HalfTensor::cat(tensors, axis);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  return output;
}

void TensorV2::print(std::ostream &out) const { itensor->print(out); }

void TensorV2::putData() const { itensor->putData(); }

void TensorV2::setData(const std::shared_ptr<MemoryData> buf, size_t off,
                       bool init) {
  itensor->setMemoryData(buf, off);

  if (buf && init) {
    initialize();
  }
}

const std::shared_ptr<MemoryData> TensorV2::getMemoryData() const {
  return itensor->getMemoryData();
}

size_t TensorV2::getOffset() const { return itensor->getOffset(); }

void TensorV2::copy(const TensorV2 &from) {
  /// @todo enable copy to non-contiguous tensor
  if (!itensor->getContiguous()) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size() &&
      getDataType() == from.getDataType()) {
    // if tensor size and data type match, copy data
    itensor->copy(from);
  } else {
    // replace with a new tensor that are the same with the given tensor
    if (from.getDataType() == ml::train::TensorDim::DataType::FP32) {
      TensorV2 t = TensorV2(from.getDim(), from.getData<float>());
      swap(t, *this);
    } else if (from.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      TensorV2 t = TensorV2(from.getDim(), from.getData<_FP16>());
      swap(t, *this);
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }
}

void TensorV2::copyData(const TensorV2 &from) { itensor->copyData(from); }

void TensorV2::copy_with_stride(const TensorV2 &from) {
  if (itensor->getDim() == from.getDim()) {
    // if the tensor dim matches, copy the data
    copy(from);
  } else {
    // replace with a new tensor that has the same data as the given tensor
    TensorV2 t = TensorV2(from.getDim(), true);
    for (unsigned int b = 0; b < t.batch(); ++b) {
      for (unsigned int c = 0; c < t.channel(); ++c) {
        for (unsigned int h = 0; h < t.height(); ++h) {
          for (unsigned int w = 0; w < t.width(); ++w) {
            if (getDataType() == ml::train::TensorDim::DataType::FP32) {
              t.setValue(b, c, h, w, from.getValue<float>(b, c, h, w));
            } else if (getDataType() == ml::train::TensorDim::DataType::FP16) {
              /// @todo remove #ifdef ENABLE_FP16
#ifdef ENABLE_FP16
              t.setValue(b, c, h, w, from.getValue<_FP16>(b, c, h, w));
#else
              throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
            }
          }
        }
      }
    }
    swap(t, *this);
  }
}

TensorV2 TensorV2::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = getDim();
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->getDim().getFeatureLen(),
                             true, "");
}

TensorV2 TensorV2::clone() const {
  TensorV2 output(getName(), getFormat(), getDataType());
  output.copy(*this);
  return output;
}

void TensorV2::save(std::ostream &file) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot save.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, getData<char>(), sz, "[Tensor::save] operation failed");
  putData();
}

void TensorV2::read(std::ifstream &file) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, getData<char>(), sz, "[Tensor::read] operation failed");
  putData();
}

std::vector<unsigned int> TensorV2::argmax() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmax.";
  return itensor->argmax();
}

float TensorV2::max_abs() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get max_abs.";
  return itensor->max_abs();
}

float TensorV2::maxValue() const { return itensor->maxValue(); }

float TensorV2::minValue() const { return itensor->minValue(); }

TensorV2 TensorV2::transpose(const std::string &direction) const {
  TensorV2 output(getDim());
  transpose(direction, output);
  return output;
}

TensorV2 &TensorV2::transpose(const std::string &direction,
                              TensorV2 &output) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (output.getData<char>() == getData<char>()) {
    TensorV2 result = clone();
    return result.transpose(direction, output);
  }

  itensor->transpose(direction, output);

  return output;
}

void TensorV2::reshape(const TensorDim &d) { itensor->reshape(d); }

void TensorV2::fill(const TensorV2 &from, bool allocate) {
  if (allocate && this->empty()) {
    this->copy(from);
    return;
  }

  if (!from.getContiguous() || !getContiguous()) {
    /// @todo enable this if needed
    throw nntrainer::exception::not_supported(
      "[Tensor::fill] non-contiguous tensors are not supported");
  }

  if (getDim() != from.getDim()) {
    throw std::invalid_argument("[Tensor::fill] dimension must be the same");
  }

  if (getStrides() != from.getStrides()) {
    /// @todo length does not represent buffer size, there should be way to
    /// get the buffer size
    throw std::invalid_argument("[Tensor::fill] buffer size must be the same");
  }

  copyData(from);
}

TensorDim TensorV2::getDim() const { return itensor->getDim(); }

TensorDim::TensorType TensorV2::getTensorType() const {
  return itensor->getTensorType();
};

Initializer TensorV2::getInitializer() const {
  return itensor->getInitializer();
}

TensorDim::Format TensorV2::getFormat() const { return itensor->getFormat(); }

Tdatatype TensorV2::getDataType() const { return itensor->getDataType(); }

void TensorV2::updateBatch(unsigned int batch) { itensor->updateBatch(batch); }

const bool TensorV2::getContiguous() const noexcept {
  return itensor->getContiguous();
}

const std::array<size_t, TensorDim::MAXDIM>
TensorV2::getStrides() const noexcept {
  return itensor->getStrides();
}

bool TensorV2::checkContinuous(unsigned int np1, unsigned int np2) const {
  if (np1 > 3 || np2 > 3) {
    throw std::invalid_argument(
      "Error: Input value must be within the range of 0 to 3.");
  }

  if (getFormat() == Tformat::NCHW) {
    if (np1 + 1 == np2)
      return true;
  } else {
    std::vector<unsigned int> continuous_order_nhwc = {0, 3, 1, 2};
    if (continuous_order_nhwc[np2] == continuous_order_nhwc[np1] + 1)
      return true;
  }

  return false;
}

void TensorV2::setName(const std::string &name_) { itensor->setName(name_); }

const std::string &TensorV2::getName() const { return itensor->getName(); }

size_t TensorV2::getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
  return itensor->getIndex(b, c, h, w);
}

size_t TensorV2::size() const { return itensor->size(); }

bool TensorV2::empty() const { return itensor->empty(); }

size_t TensorV2::bytes() const { return itensor->bytes(); }

size_t TensorV2::batch() const { return itensor->batch(); }

size_t TensorV2::channel() const { return itensor->channel(); }

size_t TensorV2::height() const { return itensor->height(); }

size_t TensorV2::width() const { return itensor->width(); }

void TensorV2::mergeAxis(unsigned int axis1, unsigned int axis2) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot merge axis";

  if (axis2 != axis1 + 1)
    if (!checkContinuous(axis1, axis2))
      throw std::invalid_argument("axis2 must be axis1 + 1 for merging.");

  itensor->mergeAxis(axis1, axis2);
}

void TensorV2::createSharedDataTensor(const TensorV2 &src, TensorV2 &dest,
                                      size_t offset) const {
  itensor->createSharedDataTensor(src.itensor.get(), dest.itensor.get(),
                                  offset);
}

TensorV2 TensorV2::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                       bool reset_stride,
                                       const std::string &name_) const {
  TensorV2 ret = *this;
  itensor->getSharedDataTensor(dim_, offset, reset_stride, name_,
                               ret.itensor.get());
  return ret;
}

void TensorV2::setTensorVar(TensorDim d, void *buf, size_t offset) {
  itensor->setTensorVar(d, buf, offset);
}

std::ostream &operator<<(std::ostream &out, TensorV2 const &input) {
  input.print(out);
  return out;
}

} // namespace nntrainer
