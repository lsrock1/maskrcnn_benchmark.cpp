#include "vision_cpu.h"


namespace rcnn{
namespace layers{

torch::Tensor box_iou_cpu(torch::Tensor area_a, torch::Tensor area_b, torch::Tensor bbox_a, torch::Tensor bbox_b){
  int TO_REMOVE = 1;
  torch::Tensor lt = torch::max(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/0, /*end=*/2), bbox_b.slice(1, 0, 2));
  torch::Tensor rb = torch::min(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/2, /*end=*/4), bbox_b.slice(1, 2, 4));
  torch::Tensor wh = (rb - lt + TO_REMOVE).clamp(0);
  torch::Tensor inter = wh.select(2, 0) * wh.select(2, 1);
  return inter / (area_a.unsqueeze(1) + area_b - inter);
}

}
}