// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

namespace rcnn{
namespace layers{
// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
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

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

class ROIAlignImpl : public torch::nn::Module {
  public:
    ROIAlignImpl(std::pair<int, int> output_size, float spatial_scale, int sampling_ratio);
    torch::Tensor forward(const at::Tensor& x, at::Tensor rois);

  private:
    int pooled_height_;
    int pooled_width_;
    float spatial_scale_;
    int sampling_ratio_;
};

TORCH_MODULE(ROIAlign);

struct ROIAlignBackward : public torch::autograd::Function{
  torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;
  void release_variables() override;

  torch::autograd::SavedVariable rois_;
  torch::IntArrayRef input_shape_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
  int sampling_ratio_;
};

}
}