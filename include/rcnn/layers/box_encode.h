#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif


namespace rcnn{
namespace layers{

torch::Tensor box_encode(torch::Tensor reference_boxes, torch::Tensor proposals, float wx, float wy, float ww, float wh) {

  if (reference_boxes.is_cuda() && proposals.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    return torch::stack(box_encode_cuda(reference_boxes, proposals, wx, wy, ww, wh), 1);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  torch::Tensor result = box_encode_cpu(reference_boxes, proposals, wx, wy, ww, wh);
  return result;
}
}//layers
}//rcnn
