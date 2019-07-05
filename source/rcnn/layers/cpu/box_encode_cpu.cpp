#include "vision_cpu.h"


namespace rcnn{
namespace layers{

torch::Tensor box_encode_cpu(torch::Tensor reference_boxes, torch::Tensor proposals, float wx, float wy, float ww, float wh){
  int TO_REMOVE = 1;
  torch::Tensor ex_widths = proposals.select(1, 2) - proposals.select(1, 0) + TO_REMOVE;
  torch::Tensor ex_heights = proposals.select(1, 3) - proposals.select(1, 1) + TO_REMOVE;
  torch::Tensor ex_ctr_x = proposals.select(1, 0) + 0.5 * ex_widths;
  torch::Tensor ex_ctr_y = proposals.select(1, 1) + 0.5 * ex_heights;

  torch::Tensor gt_widths = reference_boxes.select(1, 2) - reference_boxes.select(1, 0) + TO_REMOVE;
  torch::Tensor gt_heights = reference_boxes.select(1, 3) - reference_boxes.select(1, 1) + TO_REMOVE;
  torch::Tensor gt_ctr_x = reference_boxes.select(1, 0) + 0.5 * gt_widths;
  torch::Tensor gt_ctr_y = reference_boxes.select(1, 1) + 0.5 * gt_heights;

  return torch::stack({
    /*targets_dx*/ wx * (gt_ctr_x - ex_ctr_x) / ex_widths,
    /*targets_dy*/ wy * (gt_ctr_y - ex_ctr_y) / ex_heights,
    /*targets_dw*/ ww * torch::log(gt_widths / ex_widths),
    /*targets_dh*/ wh * torch::log(gt_heights / ex_heights)
  }, /*dim=*/1);
}

}
}