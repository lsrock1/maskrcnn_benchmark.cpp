#pragma once
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif


namespace rcnn{
namespace layers{
torch::Tensor match_proposals(torch::Tensor match_quality_matrix, bool allow_low_quality_matches, 
                                float low_th, float high_th, const bool cuda_extension=false) {

  if (match_quality_matrix.is_cuda() && cuda_extension) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    return match_proposals_cuda(match_quality_matrix, allow_low_quality_matches, low_th, high_th);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  torch::Tensor result = match_proposals_cpu(match_quality_matrix, allow_low_quality_matches, low_th, high_th);
  return result;
}
}//layers
}//rcnn
