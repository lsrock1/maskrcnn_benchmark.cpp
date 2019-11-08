#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

namespace rcnn {
namespace layers {

inline torch::Tensor box_iou(const torch::Tensor& area_a,
                             const torch::Tensor& area_b,
                             const torch::Tensor& box_a,
                             const torch::Tensor& box_b) {

  if (box_a.is_cuda() && box_b.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    return box_iou_cuda(box_a, box_b);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  torch::Tensor result = box_iou_cpu(area_a, area_b, box_a, box_b);
  return result;
}

} // namespace layers
} // namespace rcnn
