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
torch::Tensor ROIAlign_forward(const torch::Tensor& input,
                            const torch::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio);

torch::Tensor ROIAlign_backward(const torch::Tensor& grad,
                             const torch::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio);

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
