#include "rpn/inference.h"
#include "rpn/utils.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{
  
RPNPostProcessorImpl::RPNPostProcessorImpl(int pre_nms_top_n, int post_nms_top_n, float nms_thresh, int min_size, BoxCoder box_coder, int fpn_post_nms_top_n)
                                :pre_nms_top_n_(pre_nms_top_n),
                                 post_nms_top_n_(post_nms_top_n),
                                 nms_thresh_(nms_thresh),
                                 box_coder_(box_coder),
                                 fpn_post_nms_top_n_(fpn_post_nms_top_n){}

std::vector<rcnn::structures::BoxList> RPNPostProcessorImpl::AddGtProposals(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets){
  auto device = proposals[0].get_bbox().device();
  std::vector<rcnn::structures::BoxList> return_proposals;
  std::vector<rcnn::structures::BoxList> gt_boxes;
  for(auto& target: targets)
    gt_boxes.push_back(target.CopyWithFields());
  for(auto& gt_box: gt_boxes)
    gt_box.AddField("objectness", torch::ones(gt_box.Length()), torch::TensorOptions().device(device));
  for(int i = 0; i < proposals.size(); ++i){
    return_proposals.push_back(rcnn::structures::BoxList::BoxListCat(std::vector<rcnn::structures::BoxList> {proposals[i], gt_boxes[i]}));
  }
  return return_proposals;
}

std::vector<rcnn::structures::BoxList> ForwardForSingleFeatureMap(std::vector<rcnn::structures::BoxList> anchors, torch::Tensor objectness, torch::Tensor box_regression){
  auto device = objectness.device();
  int N = objectness.size(0), A = objectness.size(1), H = objectness.size(2), W = objectness.size(3);
  objectness = PermuteAndFlatten(objectness, N, A, 1, H, W).view({N, -1});
  objectness = objectness.sigmoid_();

  box_regression = PermuteAndFlatten(box_regression, N, A, 4, H, W);

  int num_anchors = A * H * W;

  int pre_nms_top_n = std::min(pre_nms_top_n_, num_anchors);
  torch::Tensor topk_idx;
  std::tie(objectness, topk_idx) = objectness.topk(pre_nms_top_n, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  box_regression = box_regression.index_select(/*dim=*/1, topk_idx);
  
  std::vector<std::pair<int64_t, int64_t>> image_shapes;
  std::vector<torch::Tensor> concat_anchors_vec;

  for(auto& box: anchors){
    image_shapes.push_back(box.get_size());
    concat_anchors_vec.puah_back(box.get_bbox());
  }
  
  torch::Tensor concat_anchors = torch::cat(concat_anchors_vec, /*dim=*/0).reshape({N, -1, 4}).index_select(1, topk_idx);
  auto proposals = box_coder_.decode(box_regression.view({-1, 4}), concat_anchors.view({-1, 4}));
  proposals = proposals.view({N, -1, 4});

  std::vector<rcnn::structures::BoxList> result;
  for(int i = 0; i < N, ++i){
    rcnn::structures::BoxList boxlist = rcnn::structures::BoxList(proposals[i], image_shapes[i], "xyxy");
    boxlist.AddField("objectness", objectness[i]);
    boxlist = boxlist.ClipToImage(false);
    boxlist = boxlist.RemoveSmallBoxes(min_size_);
    boxlist = boxlist.nms(nms_thresh_, post_nms_top_n_, "objectness");
    result.push_back(boxlist);
  }
  return result;
}

}
}