#include "roi_align.h"

namespace rcnn{
namespace layers{

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

torch::Tensor ROIAlign_backward(const torch::Tensor& grad,
                             const torch::Tensor& rois,
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

torch::autograd::variable_list ROIAlignBackward::apply(torch::autograd::variable_list&& grads) {
    // Our function had one output, so we only expect 1 gradient
  auto& grad = grads[0].data();
  auto rois = rois_.unpack();

    // Variable list to hold the gradients at the function's input variables
  torch::autograd::variable_list grad_inputs(1); 

    // Do gradient computation for each of the inputs
  if (should_compute_output(0)) {
      grad_inputs[0] = ROIAlign_backward(grad, 
                                        rois, 
                                        spatial_scale_, 
                                        pooled_height_, 
                                        pooled_width_, 
                                        input_shape_[0], 
                                        input_shape_[1], 
                                        input_shape_[2], 
                                        input_shape_[3],
                                        sampling_ratio_);
  }

  return grad_inputs;
}

void ROIAlignBackward::release_variables(){
  rois_.reset_data();
  rois_.reset_grad_function();
}

ROIAlignImpl::ROIAlignImpl(std::pair<int, int> output_size, float spatial_scale, int sampling_ratio)
                : pooled_height_(std::get<0>(output_size)),
                  pooled_width_(std::get<1>(output_size)),
                  spatial_scale_(spatial_scale),
                  sampling_ratio_(sampling_ratio){}

torch::Tensor ROIAlignImpl::forward(const torch::Tensor& x, torch::Tensor rois){
  const auto& x_ = torch::autograd::as_variable_ref(x);
  auto& rois_ = torch::autograd::as_variable_ref(rois);
  auto result = ROIAlign_forward(x_, rois_, spatial_scale_, pooled_height_, pooled_width_, sampling_ratio_);
  if(x.requires_grad()){
    auto grad_fn = std::shared_ptr<ROIAlignBackward>(new ROIAlignBackward(), torch::autograd::deleteFunction);
    grad_fn -> set_next_edges(torch::autograd::collect_next_edges(x));
    grad_fn -> rois_ = torch::autograd::SavedVariable(rois, false);
    grad_fn -> input_shape_ = x.sizes();
    grad_fn -> pooled_height_ = pooled_height_;
    grad_fn -> pooled_width_ = pooled_width_;
    grad_fn -> spatial_scale_ = spatial_scale_;
    grad_fn -> sampling_ratio_ = sampling_ratio_;
    set_history(torch::autograd::flatten_tensor_args(result), grad_fn);
  }
  return result;
}

}
}