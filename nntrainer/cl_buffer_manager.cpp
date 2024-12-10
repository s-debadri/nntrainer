// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cl_buffer_manager.h>
#include <cstring>

namespace nntrainer {

ClBufferManager &ClBufferManager::getInstance() {
  static ClBufferManager instance;
  return instance;
}

// to-do: Implementation to be updated with array of Buffer objects if required
// fp16 Buffer objects to be added in future
void ClBufferManager::initBuffers() {
  readBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  readBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  readBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  writeBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  writeBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  ml_logi("ClBufferManager: Buffers initialized");
}

bool ClBufferManager::writeHost(const float *data, size_t size, opencl::Buffer* buff){
  float *hostMem = (float *)(buff->MapBuffer(command_queue_inst_, 0,
                                          buffer_size_bytes, false));
  std::memcpy(hostMem, data, size);
  // signal GPU that host is done accessing memory
  bool result = buff->UnMapBuffer(command_queue_inst_, hostMem);
  return result;
}

bool ClBufferManager::readHost(float *data, size_t size, opencl::Buffer* buff){
  // to avoid cache inconsistency
  float *hostMem = (float *)(buff->MapBuffer(command_queue_inst_, 0,
                                          buffer_size_bytes, false));
  std::memcpy(data, hostMem, size);

  bool result = buff->UnMapBuffer(command_queue_inst_, hostMem);
  return result;
}

ClBufferManager::~ClBufferManager() {
  delete readBufferA;
  delete readBufferB;
  delete readBufferC;
  delete writeBufferA;
  delete writeBufferB;

  ml_logi("ClBufferManager: Buffers destroyed");
}

} // namespace nntrainer
