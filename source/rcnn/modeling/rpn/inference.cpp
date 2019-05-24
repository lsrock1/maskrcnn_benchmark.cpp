#include "rpn/inference.h"
#include "rpn/utils.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{
  
RPNPostProcessorImpl::RPNPostProcessorImpl(const int64_t pre_nms_top_n, const int64_t post_nms_top_n, const float nms_thresh, const int64_t min_size, BoxCoder& box_coder, const int64_t fpn_post_nms_top_n,  const bool fpn_post_nms_per_batch)
                                :pre_nms_top_n_(pre_nms_top_n),
                                 post_nms_top_n_(post_nms_top_n),
                                 nms_thresh_(nms_thresh),
                                 box_coder_(box_coder),
                                 fpn_post_nms_top_n_(fpn_post_nms_top_n),
                                 fpn_post_nms_per_batch_(fpn_post_nms_per_batch){}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::AddGtProposals(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets){
  auto device = proposals[0].get_bbox().device();
  std::vector<rcnn::structures::BoxList> return_proposals;
  std::vector<rcnn::structures::BoxList> gt_boxes;
  for(auto& target: targets)
    gt_boxes.push_back(target.CopyWithFields(std::vector<std::string> {}));
  for(auto& gt_box: gt_boxes)
    gt_box.AddField("objectness", torch::ones(gt_box.Length(), torch::TensorOptions().device(device)));
  for(int i = 0; i < proposals.size(); ++i){
    return_proposals.push_back(rcnn::structures::BoxList::CatBoxList(std::vector<rcnn::structures::BoxList> {proposals[i], gt_boxes[i]}));
  }
  return return_proposals;
}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::ForwardForSingleFeatureMap(std::vector<rcnn::structures::BoxList> anchors, torch::Tensor objectness, torch::Tensor box_regression){
  auto device = objectness.device();
  int N = objectness.size(0), A = objectness.size(1), H = objectness.size(2), W = objectness.size(3);
  objectness = PermuteAndFlatten(objectness, N, A, 1, H, W).view({N, -1});
  objectness = objectness.sigmoid_();

  box_regression = PermuteAndFlatten(box_regression, N, A, 4, H, W);

  int64_t num_anchors = A * H * W;

  int64_t pre_nms_top_n = std::min(pre_nms_top_n_, num_anchors);
  torch::Tensor topk_idx;
  std::tie(objectness, topk_idx) = objectness.topk(pre_nms_top_n, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  box_regression = box_regression.index_select(/*dim=*/1, topk_idx);
  
  std::vector<std::pair<int64_t, int64_t>> image_shapes;
  std::vector<torch::Tensor> concat_anchors_vec;

  for(auto& box: anchors){
    image_shapes.push_back(box.get_size());
    concat_anchors_vec.push_back(box.get_bbox());
  }
  
  torch::Tensor concat_anchors = torch::cat(concat_anchors_vec, /*dim=*/0).reshape({N, -1, 4}).index_select(1, topk_idx);
  auto proposals = box_coder_.decode(box_regression.view({-1, 4}), concat_anchors.view({-1, 4}));
  proposals = proposals.view({N, -1, 4});

  std::vector<rcnn::structures::BoxList> result;
  assert(proposals.size(0) == objectness.size(0));
  for(int i = 0; i < N; ++i){
    rcnn::structures::BoxList boxlist = rcnn::structures::BoxList(proposals[i], image_shapes[i], "xyxy");
    boxlist.AddField("objectness", objectness[i]);
    boxlist = boxlist.ClipToImage(false);
    boxlist = boxlist.RemoveSmallBoxes(min_size_);
    boxlist = boxlist.nms(nms_thresh_, post_nms_top_n_, "objectness");
    result.push_back(boxlist);
  }
  return result;
}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::forward(std::vector<std::vector<rcnn::structures::BoxList>> anchors, std::vector<torch::Tensor> objectness, std::vector<torch::Tensor> box_regression){
  std::vector<std::vector<rcnn::structures::BoxList>> sampled_boxes;//{feature1{image1, image2 ...}, feature2 ...}
  int num_levels = objectness.size();//== num_feature_maps

  for(int i = 0; i < num_levels; ++i){
    std::vector<rcnn::structures::BoxList> boxes_per_feature_map;
    std::for_each(anchors.begin(), anchors.end(), [&](std::vector<rcnn::structures::BoxList> anchors_per_image){boxes_per_feature_map.push_back(anchors_per_image[i]);});
    sampled_boxes.push_back(ForwardForSingleFeatureMap(boxes_per_feature_map, objectness[i], box_regression[i]));
  }

  std::vector<std::vector<rcnn::structures::BoxList>> boxlists;//image{features }
  for(int i = 0; i < sampled_boxes[0].size(); ++i){//iterate num_images
    std::vector<rcnn::structures::BoxList> boxlist;
    std::for_each(sampled_boxes.begin(), sampled_boxes.end(), [&](std::vector<rcnn::structures::BoxList> boxes_per_feature_map){boxlist.push_back(boxes_per_feature_map[i]);});
    boxlists.push_back(boxlist);
  }

  std::vector<rcnn::structures::BoxList> return_boxlists;
  for(auto& boxlist: boxlists){
    return_boxlists.push_back(rcnn::structures::BoxList::CatBoxList(boxlist));
  }

  if(num_levels > 1){
    return_boxlists = SelectOverAllLayers(return_boxlists);
  }
  return return_boxlists;
}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::forward(std::vector<std::vector<rcnn::structures::BoxList>> anchors, std::vector<torch::Tensor> objectness, std::vector<torch::Tensor> box_regression, std::vector<rcnn::structures::BoxList> targets){
  std::vector<rcnn::structures::BoxList> boxlists = forward(anchors, objectness, box_regression);
  if(is_training())
    return AddGtProposals(boxlists, targets);
}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::SelectOverAllLayers(std::vector<rcnn::structures::BoxList> boxlists){
  int num_images = boxlists.size();

  if(is_training() && fpn_post_nms_per_batch_){
    //to all images
    std::vector<torch::Tensor> objectness_vec;
    std::vector<int64_t> box_sizes;
    for(auto& boxlist: boxlists){
      objectness_vec.push_back(boxlist.GetField("objectness"));
      box_sizes.push_back(boxlist.Length());
    }
    torch::Tensor objectness = torch::cat(objectness_vec, 0);
    
    int64_t post_nms_top_n = std::min(fpn_post_nms_top_n_, objectness.size(0));
    torch::Tensor inds_sorted = std::get<1>(torch::topk(objectness, post_nms_top_n, 0, /*largest*/true, /*sorted=*/true));
    torch::Tensor inds_mask = torch::zeros_like(objectness, torch::TensorOptions().dtype(torch::kUInt8));
    inds_mask.index_fill_(0, inds_sorted, 1);

    std::vector<torch::Tensor> inds_mask_vec = inds_mask.split_with_sizes(box_sizes, /*dim=*/0);
    for(int i = 0; i < num_images; ++i){
      boxlists[i] = boxlists[i][inds_mask_vec[i]];
    }
  }
  else{
    //to each image
    for(int i = 0; i < num_images; ++i){
      torch::Tensor objectness = boxlists[i].GetField("objectness");
      int64_t post_nms_top_n = std::min(fpn_post_nms_top_n_, objectness.size(0));
      torch::Tensor inds_sorted = std::get<1>(torch::topk(objectness, post_nms_top_n, 0, /*largest*/true, /*sorted=*/true));
      boxlists[i] = boxlists[i][inds_sorted];
    }
  }
  return boxlists;
}

RPNPostProcessor MakeRPNPostprocessor(BoxCoder& rpn_box_coder, bool is_train){
  std::string phase;
  
  if(is_train){
    phase = "TRAIN";
  }
  else{
    phase = "TEST";
  }
  
  int64_t fpn_post_nms_top_n = rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", (std::string("FPN_POST_NMS_TOP_N_") + phase).c_str()});
  int64_t pre_nms_top_n = rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", (std::string("PRE_NMS_TOP_N_") + phase).c_str()});
  int64_t post_nms_top_n = rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", (std::string("POST_NMS_TOP_N_") + phase).c_str()});

  return RPNPostProcessor(
    pre_nms_top_n,
    post_nms_top_n,
    rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", "NMS_THRESH"}),
    rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", "MIN_SIZE"}),
    rpn_box_coder,
    fpn_post_nms_top_n,
    rcnn::config::GetCFG<bool>({"MODEL", "RPN", "FPN_POST_NMS_PER_BATCH"})
  );
}

}
}