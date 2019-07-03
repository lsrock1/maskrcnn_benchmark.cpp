#include "roi_heads/box_head/inference.h"
#include "defaults.h"
#include <iostream>


namespace rcnn{
namespace modeling{

PostProcessorImpl::PostProcessorImpl(float score_thresh,
                             float nms, 
                             int64_t detections_per_img, 
                             BoxCoder& box_coder, 
                             bool cls_agnostic_bbox_reg , 
                             bool bbox_aug_enabled)
                             :score_thresh_(score_thresh),
                              nms_(nms),
                              detections_per_img_(detections_per_img),
                              box_coder_(box_coder),
                              cls_agnostic_bbox_reg_(cls_agnostic_bbox_reg),
                              bbox_aug_enabled_(bbox_aug_enabled){}

std::vector<rcnn::structures::BoxList> PostProcessorImpl::forward(std::pair<torch::Tensor, torch::Tensor> x, std::vector<rcnn::structures::BoxList> boxes){
  torch::Tensor class_logits, box_regression, class_prob, concat_boxes;
  std::tie(class_logits, box_regression) = x;
  class_prob = torch::softmax(class_logits, -1);

  std::vector<std::pair<int64_t, int64_t>> image_shapes;
  image_shapes.reserve(boxes.size());
  std::vector<int64_t> boxes_per_image;
  boxes_per_image.reserve(boxes.size());
  std::vector<torch::Tensor> concat_boxes_vec;
  concat_boxes_vec.reserve(boxes.size());
  for(auto& box: boxes){
    image_shapes.push_back(box.get_size());
    boxes_per_image.push_back(box.Length());
    concat_boxes_vec.push_back(box.get_bbox());
  }
  concat_boxes = torch::cat(concat_boxes_vec, /*dim=*/0);
  
  if(cls_agnostic_bbox_reg_)
    box_regression = box_regression.slice(/*dim=*/1, /*start=*/-4);

  torch::Tensor proposals = box_coder_.decode(box_regression.view({std::accumulate(boxes_per_image.begin(), boxes_per_image.end(), 0), -1}), concat_boxes);
  if(cls_agnostic_bbox_reg_)
    proposals = proposals.repeat({1, class_prob.size(1)});

  int64_t num_classes = class_prob.size(1);

  std::vector<torch::Tensor> proposals_per_img = proposals.split_with_sizes(boxes_per_image);
  std::vector<torch::Tensor> class_prob_per_img = class_prob.split_with_sizes(boxes_per_image);
  
  std::vector<rcnn::structures::BoxList> results;
  results.reserve(proposals_per_img.size());

  for(size_t i = 0; i < proposals_per_img.size(); ++i){
    rcnn::structures::BoxList boxlist = prepare_boxlist(proposals_per_img[i], class_prob_per_img[i], image_shapes[i]);
    boxlist = boxlist.ClipToImage(false);
    if(!bbox_aug_enabled_)
      boxlist = filter_results(boxlist, num_classes);
    results.push_back(boxlist);
  }

  return results;
}

rcnn::structures::BoxList PostProcessorImpl::prepare_boxlist(torch::Tensor boxes, torch::Tensor scores, std::pair<int64_t, int64_t> image_shape){
  boxes = boxes.reshape({-1, 4});
  scores = scores.reshape({-1});
  rcnn::structures::BoxList boxlist = rcnn::structures::BoxList(boxes, image_shape, "xyxy");
  boxlist.AddField("scores", scores);
  return boxlist;
}

rcnn::structures::BoxList PostProcessorImpl::filter_results(rcnn::structures::BoxList boxlist, int num_classes){
  torch::Tensor boxes = boxlist.get_bbox().reshape({-1, num_classes * 4});
  torch::Tensor scores = boxlist.GetField("scores").reshape({-1, num_classes});
  auto device = scores.device();
  std::vector<rcnn::structures::BoxList> results_vec;
  results_vec.reserve(num_classes);
  torch::Tensor inds_all = scores > score_thresh_;

  for(size_t i = 1; i < num_classes; ++i){
    torch::Tensor inds = inds_all.select(1, i).nonzero().squeeze(1);
    torch::Tensor scores_i = scores.index_select(0, inds).select(1, i);
    torch::Tensor boxes_i = boxes.index_select(0, inds).slice(1, i * 4, (i + 1) * 4);
    rcnn::structures::BoxList boxlist_for_class = rcnn::structures::BoxList(boxes_i, boxlist.get_size(), "xyxy");
    boxlist_for_class.AddField("scores", scores_i);
    boxlist_for_class = boxlist_for_class.nms(nms_);
    int64_t num_labels = boxlist_for_class.Length();
    boxlist_for_class.AddField("labels", torch::full({num_labels}, static_cast<int>(i), torch::TensorOptions().dtype(torch::kInt64).device(device)));
    results_vec.push_back(boxlist_for_class);
  }

  rcnn::structures::BoxList results = rcnn::structures::BoxList::CatBoxList(results_vec);
  int64_t number_of_detections = results.Length();

  if(number_of_detections > detections_per_img_ && detections_per_img_ > 0){
    torch::Tensor cls_scores = results.GetField("scores");
    torch::Tensor image_thresh, indice;
    std::tie(image_thresh, indice) = torch::kthvalue(cls_scores.cpu(), number_of_detections - detections_per_img_ + 1);
    torch::Tensor keep = cls_scores >= image_thresh.item<float>();
    keep = torch::nonzero(keep).squeeze(1);
    results = results[keep];
  }
  
  return results;
}

PostProcessor MakeROIBoxPostProcessor(){
  bool use_fpn = rcnn::config::GetCFG<bool>({"MODEL", "ROI_HEADS", "USE_FPN"});
  std::vector<float> bbox_reg_weights = rcnn::config::GetCFG<std::vector<float>>({"MODEL", "ROI_HEADS", "BBOX_REG_WEIGHTS"});
  BoxCoder box_coder{bbox_reg_weights};

  float score_thresh = rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "SCORE_THRESH"});
  float nms_thresh = rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "NMS"});
  int64_t detections_per_img = rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_HEADS", "DETECTIONS_PER_IMG"});
  bool cls_agnostic_bbox_reg = rcnn::config::GetCFG<bool>({"MODEL", "CLS_AGNOSTIC_BBOX_REG"});
  bool bbox_aug_enabled = rcnn::config::GetCFG<bool>({"TEST", "BBOX_AUG", "ENABLED"});
  PostProcessor postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    );
  return postprocessor;
}

}
}