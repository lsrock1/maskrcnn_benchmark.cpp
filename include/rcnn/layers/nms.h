// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

namespace rcnn{
namespace layers{
torch::Tensor nms(const torch::Tensor& dets,
               const torch::Tensor& scores,
               const float threshold) {

  if (dets.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return torch::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = torch::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}
}//layers
}//rcnn
