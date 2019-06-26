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


torch::Tensor nms_cpu(const torch::Tensor& dets,
                   const torch::Tensor& scores,
                   const float threshold);
}
}

