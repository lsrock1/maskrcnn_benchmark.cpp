// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace layers {

at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned);

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned);

torch::Tensor nms_cuda(const torch::Tensor boxes, float nms_overlap_thresh);

torch::Tensor compute_flow_cuda(const torch::Tensor& boxes,
                             const int height,
                             const int width);

std::vector<torch::Tensor> box_encode_cuda(torch::Tensor boxes, torch::Tensor anchors, float wx, float wy, float ww, float wh);

torch::Tensor box_iou_cuda(torch::Tensor box1, torch::Tensor box2);

torch::Tensor match_proposals_cuda(torch::Tensor match_quality_matrix, bool allow_low_quality_matches,
                                   float low_th, float high_th);
} // namespace layers
} // namespace rcnn
