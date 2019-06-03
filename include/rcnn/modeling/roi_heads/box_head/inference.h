#pragma once
#include <torch/torch.h>
#include "box_coder.h"
#include "bounding_box.h"


namespace rcnn{
namespace modeling{

class PostProcessorImpl : public torch::nn::Module{
  public:
    PostProcessorImpl(float score_thresh, float nms, int64_t detections_per_img, BoxCoder& box_coder, bool cls_agnostic_bbox_reg = false, bool bbox_aug_enabled = false);
    std::vector<rcnn::structures::BoxList> forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> boxes);
    rcnn::structures::BoxList prepare_boxlist(torch::Tensor boxes, torch::Tensor socres, std::pair<int64_t, int64_t> image_shape);
    rcnn::structures::BoxList filter_results(rcnn::structures::BoxList boxlist, int num_classes);

  private:
    float score_thresh_;
    float nms_;
    int64_t detections_per_img_;
    BoxCoder box_coder_;
    bool cls_agnostic_bbox_reg_;
    bool bbox_aug_enabled_;
};

TORCH_MODULE(PostProcessor);

PostProcessor MakeROIBoxPostProcessor();

}
}