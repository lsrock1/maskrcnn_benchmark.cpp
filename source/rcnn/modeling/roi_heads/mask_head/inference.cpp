#include "roi_heads/mask_head/inference.h"
#include <algorithm>
#include <cassert>
#include "mask.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{

torch::Tensor ExpandBoxes(torch::Tensor& boxes, float scale){
  torch::Tensor w_half = (boxes.select(1, 2) - boxes.select(1, 0)) * 0.5;
  torch::Tensor h_half = (boxes.select(1, 3) - boxes.select(1, 1)) * 0.5;
  torch::Tensor x_c = (boxes.select(1, 2) + boxes.select(1, 0)) * 0.5;
  torch::Tensor y_c = (boxes.select(1, 3) + boxes.select(1, 1)) * 0.5;

  w_half = w_half * scale;
  h_half = h_half * scale;

  torch::Tensor boxes_exp = torch::zeros_like(boxes);
  boxes_exp.select(1, 0) = x_c - w_half;
  boxes_exp.select(1, 2) = x_c + w_half;
  boxes_exp.select(1, 1) = y_c - h_half;
  boxes_exp.select(1, 3) = y_c + h_half;
  return boxes_exp;
}

std::pair<torch::Tensor, float> ExpandMasks(torch::Tensor mask, int padding){
  int64_t N = mask.size(0);
  int64_t M = mask.size(-1);
  int pad2 = 2 * padding;
  float scale = static_cast<float>(M + pad2) / M;
  torch::Tensor padded_mask = torch::zeros({N, 1, M + pad2, M + pad2}, mask.options());
  
  padded_mask.slice(2, padding, -padding).slice(3, padding, -padding) = mask;
  return std::make_pair(padded_mask, scale);
}

torch::Tensor PasteMaskInImage(torch::Tensor mask, torch::Tensor box, int64_t im_h, int64_t im_w, float threshold, int padding){
  mask = mask.to(torch::kF32);
  box = box.to(torch::kF32);

  torch::Tensor padded_mask;
  float scale;
  std::tie(padded_mask, scale) = ExpandMasks(mask.unsqueeze(0), padding);
  mask = padded_mask.select(0, 0).select(0, 0);
  box = ExpandBoxes(box.unsqueeze_(0), scale).select(0, 0);
  box = box.to(torch::kI32);

  int TO_REMOVE = 1;
  int64_t w = (box.select(0, 2) - box.select(0, 0) + TO_REMOVE).item<int64_t>();
  int64_t h = (box.select(0, 3) - box.select(0, 1) + TO_REMOVE).item<int64_t>();
  w = std::max(w, static_cast<int64_t> (1));
  h = std::max(h, static_cast<int64_t> (1));

  mask = mask.expand({1, 1, -1, -1});
  mask = mask.to(torch::kF32);
  mask = rcnn::layers::interpolate(mask, {h, w});
  mask = mask.select(0, 0).select(0, 0);

  if(threshold >= 0)
    mask = mask > threshold;
  else
    mask = (mask * 255).to(torch::kU8);

  torch::Tensor im_mask = torch::zeros({im_h, im_w}).to(torch::kU8);
  int x_0 = std::max(box.select(0, 0).item<int>(), 0);
  int x_1 = std::min(box.select(0, 2).item<int>() + 1, static_cast<int>(im_w));
  int y_0 = std::max(box.select(0, 1).item<int>(), 0);
  int y_1 = std::min(box.select(0, 3).item<int>() + 1, static_cast<int>(im_h));

  im_mask.slice(0, y_0, y_1).slice(1, x_0, x_1) = mask.slice(0, (y_0 - box.select(0, 1).item<int>()), (y_1 - box.select(0, 1).item<int>()))
                                                      .slice(1, (x_0 - box.select(0, 0).item<int>()), (x_1 - box.select(0, 0).item<int>()));
  return im_mask;
}

Masker::Masker(float threshold, int padding) :threshold_(threshold), padding_(padding){}

torch::Tensor Masker::ForwardSingleImage(torch::Tensor& masks, rcnn::structures::BoxList& boxes){
  boxes = boxes.Convert("xyxy");
  int64_t im_w, im_h;
  std::tie(im_w, im_h) = boxes.get_size();
  std::vector<torch::Tensor> res;
  torch::Tensor res_tensor;

  for(int i = 0; i < masks.size(0); ++i)
    res.push_back(PasteMaskInImage(masks.select(0, i).select(0, 0), boxes.get_bbox().select(0, i), im_h, im_w, threshold_, padding_));

  if(res.size() > 0)
    res_tensor = torch::stack(res, 0).unsqueeze(1);
  else
    res_tensor = torch::empty({0, 1, masks.size(-2), masks.size(-1)}, masks.options());
  return res_tensor;
}

std::vector<torch::Tensor> Masker::operator()(std::vector<torch::Tensor>& masks, std::vector<rcnn::structures::BoxList>& boxes){
  assert(masks.size() == boxes.size());

  std::vector<torch::Tensor> results;

  for(int i = 0; i < masks.size(); ++i){
    assert(masks[i].size(0) == boxes[i].Length());
    results.push_back(ForwardSingleImage(masks[i], boxes[i]));
  }
  return results;
}

MaskPostProcessorImpl::MaskPostProcessorImpl(Masker* masker) :masker_(masker){}

MaskPostProcessorImpl::~MaskPostProcessorImpl(){
  if(masker_)
    delete masker_;
}

std::vector<rcnn::structures::BoxList> MaskPostProcessorImpl::forward(torch::Tensor& x, std::vector<rcnn::structures::BoxList>& boxes){
  torch::Tensor mask_prob = x.sigmoid();
  int64_t num_masks = x.size(0);
  std::vector<torch::Tensor> labels_vec, mask_prob_vec;
  torch::Tensor labels_tensor;
  std::vector<int64_t> boxes_per_image;
  std::vector<rcnn::structures::BoxList> results;
  
  for(auto& box : boxes){
    labels_vec.push_back(box.GetField("labels"));
    boxes_per_image.push_back(box.Length());
  }
  labels_tensor = torch::cat(labels_vec);

  //torch::Tensor index = torch::arange(num_masks).to(labels_tensor.device());
  mask_prob = mask_prob.index_select(1, labels_tensor).unsqueeze(1);
  mask_prob_vec = mask_prob.split_with_sizes(boxes_per_image);

  if(masker_)
    mask_prob_vec = (*masker_)(mask_prob_vec, boxes);

  for(int i = 0; i < boxes.size(); ++i){
    auto bbox = boxes[i].Convert("xyxy");
    bbox.AddField("mask", mask_prob_vec[i]);
    results.push_back(bbox);
  }
  return results;
}

std::vector<rcnn::structures::BoxList> MaskPostProcessorCOCOFormatImpl::forward(torch::Tensor& x, std::vector<rcnn::structures::BoxList>& boxes){
  std::vector<rcnn::structures::BoxList> results = MaskPostProcessorImpl::forward(x, boxes);
  for(auto& result : results){
    torch::Tensor masks = result.GetField("mask").cpu();
    std::vector<coco::RLEstr> rles;
    for(int i = 0; i < masks.size(0); ++i){
      coco::byte* mask_array = new coco::byte[masks.numel()];
      torch::Tensor masks_reshape = masks.reshape({-1});
      for(int i = 0; i < masks_reshape.numel(); ++i)
        mask_array[i] = static_cast<coco::byte>(masks_reshape[i].item<float>());
      rles.push_back(coco::encode(mask_array, masks.size(1), masks.size(2), masks.size(0))[0]);
      delete[] mask_array;
    }

    // for(auto& rle : rles)
    result.AddField("mask", rles);
  }

  return results;
}

MaskPostProcessor MakeRoiMaskPostProcessor(){
  bool postprocess_masks = rcnn::config::GetCFG<bool>({"MODEL", "ROI_MASK_HEAD", "POSTPROCESS_MASKS"});
  if(postprocess_masks){
    Masker* masker = new Masker(rcnn::config::GetCFG<float>({"MODEL", "ROI_MASK_HEAD", "POSTPROCESS_MASKS_THRESHOLD"}), 1);
    return MaskPostProcessor(masker);
  }
  else
    return MaskPostProcessor();
}

}//modeling
}//rcnn