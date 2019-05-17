#include "rpn/utils.h"


namespace rcnn{
namespace modeling{

torch::Tensor PermuteAndFlatten(torch::Tensor layer, int64_t N, int64_t A. int64_t C, int64_t H, int64_t W){
  layer = layer.view({N, -1, C, H, W});
  layer = layer.permute(0, 3, 4, 1, 2);
  layer = layer.reshape({N, -1, C});
  return layer;
}

}
}