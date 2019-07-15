#include "rpn/utils.h"
#include <cassert>
#include <iostream>

#include <cat.h>


namespace rcnn{
namespace modeling{

torch::Tensor PermuteAndFlatten(torch::Tensor& layer, int64_t N, int64_t A, int64_t C, int64_t H, int64_t W){
  layer = layer.view({N, -1, C, H, W});
  layer = layer.permute({0, 3, 4, 1, 2});
  layer = layer.reshape({N, -1, C});
  return layer;
}

std::pair<torch::Tensor, torch::Tensor> ConcatBoxPredictionLayers(std::vector<torch::Tensor>& box_cls, std::vector<torch::Tensor>& box_regression){
  std::vector<torch::Tensor> box_cls_flattened;
  box_cls_flattened.reserve(box_cls.size());
  std::vector<torch::Tensor> box_regression_flattened;
  box_regression_flattened.reserve(box_cls.size());
  int64_t C;
  assert(box_cls.size() == box_regression.size());

  for(int i = 0; i < box_cls.size(); ++i){
    int64_t N = box_cls[i].size(0), AxC = box_cls[i].size(1), H = box_cls[i].size(2), W = box_cls[i].size(3);
    int64_t Ax4 = box_regression[i].size(1);
    int64_t A = static_cast<int64_t> (Ax4 / 4);
    C = static_cast<int64_t> (AxC / A);

    //box_cls_per_level
    box_cls_flattened.push_back(
      PermuteAndFlatten(box_cls[i], N, A, C, H, W)
    );

    box_regression_flattened.push_back(
      PermuteAndFlatten(box_regression[i], N, A, 4, H, W)
    );
  }

  return std::make_pair(rcnn::layers::cat(box_cls_flattened, /*dim=*/1).reshape({-1, C}),
                 rcnn::layers::cat(box_regression_flattened, /*dim=*/1).reshape({-1, 4}));
}

}
}