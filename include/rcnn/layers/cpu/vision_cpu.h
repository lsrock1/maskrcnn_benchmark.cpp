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

torch::Tensor box_iou_cpu(torch::Tensor area_a, torch::Tensor area_b, torch::Tensor bbox_a, torch::Tensor bbox_b);

torch::Tensor box_encode_cpu(torch::Tensor reference_boxes, torch::Tensor proposals, float wx, float wy, float ww, float wh);

torch::Tensor match_proposals_cpu(torch::Tensor match_quality_matrix, bool allow_low_quality_matches, 
                                float low_th, float high_th);
}
}
