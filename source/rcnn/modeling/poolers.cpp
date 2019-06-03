#include "poolers.h"
#include "cat.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{
  
LevelMapper::LevelMapper(int k_min, int k_max, int canonical_scale, int canonical_level, float eps)
                        :k_min_(k_min),
                         k_max_(k_max),
                         s0_(canonical_scale),
                         lvl0_(canonical_level),
                         eps_(eps){}

torch::Tensor LevelMapper::operator()(std::vector<rcnn::structures::BoxList> boxlists){
  torch::Tensor s;
  std::vector<torch::Tensor> area_vec;
  for(auto& boxlist: boxlists){
    area_vec.push_back(boxlist.Area());
  }
  s = rcnn::layers::cat(area_vec, 0).sqrt_();
  torch::Tensor target_lvls = torch::floor(lvl0_ + torch::log2(s / s0_ + eps_));
  target_lvls = torch::clamp(target_lvls, /*min=*/k_min_, /*max=*/k_max_);
  return target_lvls.to(torch::kI64) - k_min_;
}

PoolerImpl::PoolerImpl(std::pair<int, int> output_size, std::vector<float> scales, int sampling_ratio)
              :output_size_(output_size),
               map_levels_(LevelMapper(-torch::log2(torch::tensor(scales[0], torch::TensorOptions().dtype(torch::kF32))).item<int>(),
                                       -torch::log2(torch::tensor(scales.back(), torch::TensorOptions().dtype(torch::kF32))).item<int>())){
  for(int i = 0; i < scales.size(); ++i){
    poolers_.push_back(register_module("pooler" + std::to_string(i+1), rcnn::layers::ROIAlign(output_size_, scales[i], sampling_ratio)));
  }
}

torch::Tensor PoolerImpl::ConvertToROIFormat(std::vector<rcnn::structures::BoxList> boxes){
  std::vector<torch::Tensor> concat_boxes_vec;
  std::vector<torch::Tensor> ids_vec;
  auto device = boxes[0].get_bbox().device();
  auto dtype = boxes[0].get_bbox().dtype();
  for(int i = 0; i < boxes.size(); ++i){
    concat_boxes_vec.push_back(boxes[i].Area());
    ids_vec.push_back(torch::full({boxes[i].Length(), 1}, i, torch::TensorOptions().dtype(dtype).device(device)));
  }
  torch::Tensor concat_boxes = rcnn::layers::cat(concat_boxes_vec, 0);
  torch::Tensor ids = rcnn::layers::cat(ids_vec, 0);
  return torch::cat({ids, concat_boxes}, 1);
}

torch::Tensor PoolerImpl::forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> boxes){
  int num_levels = poolers_.size();
  torch::Tensor rois = ConvertToROIFormat(boxes);
  if(num_levels == 1)
    return poolers_[0](x[0], rois);

  torch::Tensor levels = map_levels_(boxes);
  int num_rois = rois.size(0);
  int num_channels = x[0].size(1);
  int output_size = std::get<0>(output_size_);

  auto dtype = x[0].dtype();
  auto device = x[0].device();
  torch::Tensor result = torch::zeros({num_rois, num_channels, output_size, output_size}, 
                                torch::TensorOptions().dtype(dtype).device(device));

  for(int level = 0; level < boxes.size(); ++level){
    torch::Tensor idx_in_level = torch::nonzero(levels == level).squeeze(1);
    torch::Tensor rois_per_level = rois.index_select(/*dim=*/0, idx_in_level);
    result.index_copy_(/*dim=*/0, idx_in_level, poolers_[level]->forward(/*per_level_feature=*/x[level], rois_per_level));
  }

  return result;
}

Pooler MakePooler(std::string head_name){
  int resolution = rcnn::config::GetCFG<int>({"MODEL", head_name.c_str(), "POOLER_RESOLUTION"});
  std::vector<float> scales = rcnn::config::GetCFG<std::vector<float>>({"MODEL", head_name.c_str(), "POOLER_SCALES"});
  int sampling_ratio = rcnn::config::GetCFG<int>({"MODEL", head_name.c_str(), "POOLER_SAMPLING_RATIO"});
  return Pooler(std::make_pair(resolution, resolution), scales, sampling_ratio);
}

}
}