// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/torch.h>

namespace rcnn{
namespace layers{
torch::Tensor ROIAlign_forward_cpu(const torch::Tensor& input,
                                const torch::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);


// torch::Tensor ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);


// Interface for Python
torch::Tensor ROIAlign_forward(const torch::Tensor& input,
                               const torch::Tensor& rois,
                               const float spatial_scale,
                               const int pooled_height,
                               const int pooled_width,
                               const int sampling_ratio) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_forward_cpu(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
}

// at::Tensor ROIAlign_backward(const at::Tensor& grad,
//                              const at::Tensor& rois,
//                              const float spatial_scale,
//                              const int pooled_height,
//                              const int pooled_width,
//                              const int batch_size,
//                              const int channels,
//                              const int height,
//                              const int width,
//                              const int sampling_ratio) {
//   if (grad.type().is_cuda()) {
// #ifdef WITH_CUDA
//     return ROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio);
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }
//   AT_ERROR("Not implemented on the CPU");
// }
}//layers
}//rcnn