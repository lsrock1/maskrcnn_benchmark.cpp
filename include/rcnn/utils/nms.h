#pragma once
#include <torch/torch.h>

// torch::Tensor ROIAlign_forward_cuda(const torch::Tensor& input,
//                                  const torch::Tensor& rois,
//                                  const float spatial_scale,
//                                  const int pooled_height,
//                                  const int pooled_width,
//                                  const int sampling_ratio);

// at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
//                                   const at::Tensor& rois,
//                                   const float spatial_scale,
//                                   const int pooled_height,
//                                   const int pooled_width,
//                                   const int batch_size,
//                                   const int channels,
//                                   const int height,
//                                   const int width,
//                                   const int sampling_ratio);


torch::Tensor ROIAlign_forward_cpu(const torch::Tensor& input,
                                const torch::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);


// torch::Tensor nms_cuda(const torch::Tensor boxes, float nms_overlap_thresh);


torch::Tensor nms_cpu(const torch::Tensor& dets,
                   const torch::Tensor& scores,
                   const float threshold);


torch::Tensor nms(const torch::Tensor& dets,
               const torch::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return torch::empty({0}, dets.options().dtype(torch::kLong).device(torch::kCPU));
    auto b = torch::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  torch::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}