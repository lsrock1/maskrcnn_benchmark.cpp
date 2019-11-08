// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/torch.h>
#include <cpu/vision_cpu.h>

#ifdef WITH_CUDA
#include <cuda/vision_cuda.h>
#endif

using namespace torch::autograd;

namespace rcnn {
namespace layers {

class _ROIAlign : public torch::autograd::Function<_ROIAlign> {

public:
  static inline Variable forward(AutogradContext *ctx,
                                 const Variable& input,
                                 const Variable& roi,
                                 const int64_t output_size_height,
                                 const int64_t output_size_width,
                                 const double spatial_scale,
                                 const int64_t sampling_ratio) {
    ctx->save_for_backward({roi});
    ctx->saved_data["output_size_height"] = output_size_height;
    ctx->saved_data["output_size_width"] = output_size_width;
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["sampling_ratio"] = sampling_ratio;
    ctx->saved_data["input_shape"] = input.sizes();

    if (input.type().is_cuda()) {
  #ifdef WITH_CUDA
      return ROIAlign_forward_cuda(input, roi, spatial_scale, output_size_height, output_size_width, sampling_ratio);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
    }
    return ROIAlign_forward_cpu(input, roi, spatial_scale, output_size_height, output_size_width, sampling_ratio);
  }

  static inline variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto rois = ctx->get_saved_variables()[0];
    int64_t output_size_height = ctx->saved_data["output_size_height"].toInt();
    int64_t output_size_width = ctx->saved_data["output_size_width"].toInt();
    double spatial_scale = ctx->saved_data["spatial_scale"].toDouble();
    int64_t sampling_ratio = ctx->saved_data["sampling_ratio"].toInt();
    std::vector<int64_t> input_shape = ctx->saved_data["input_shape"].toIntListRef().vec();
    torch::Tensor grad_input;
    auto grad = grad_outputs[0];

    if (grad.type().is_cuda()) {
  #ifdef WITH_CUDA
      grad_input = ROIAlign_backward_cuda(grad, rois, spatial_scale, output_size_height, output_size_width, input_shape[0], input_shape[1], input_shape[2], input_shape[3], sampling_ratio);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
    }
    AT_ERROR("Not implemented on the CPU");
    return variable_list{grad_input, Variable(), Variable(), Variable(), Variable()};
  }
};

class ROIAlignImpl : public torch::nn::Module {

public:
  ROIAlignImpl(std::pair<int64_t, int64_t> output_size, double spatial_scale, int64_t sampling_ratio);
  torch::Tensor forward(const torch::Tensor& x, torch::Tensor rois);
  std::shared_ptr<ROIAlignImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  int64_t pooled_height_;
  int64_t pooled_width_;
  double spatial_scale_;
  int64_t sampling_ratio_;
};

TORCH_MODULE(ROIAlign);

} // namespace layers
} // namespace rcnn
