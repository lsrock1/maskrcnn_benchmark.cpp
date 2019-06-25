#include "transforms/build.h"
#include "defaults.h"
#include <torch/torch.h>

namespace rcnn{
namespace data{

Compose BuildTransforms(bool is_train){
  int min_size, max_size;
  float flip_horizontal_prob, flip_vertical_prob;
  if(is_train){
    min_size = rcnn::config::GetCFG<int>({"INPUT", "MIN_SIZE_TRAIN"});
    max_size = rcnn::config::GetCFG<int>({"INPUT", "MAX_SIZE_TRAIN"});
    flip_horizontal_prob = 0.5;
    flip_vertical_prob = rcnn::config::GetCFG<int>({"INPUT", "VERTICAL_FLIP_PROB_TRAIN"});
  }
  else{
    min_size = rcnn::config::GetCFG<int>({"INPUT", "MIN_SIZE_TEST"});
    max_size = rcnn::config::GetCFG<int>({"INPUT", "MAX_SIZE_TEST"});
    flip_horizontal_prob = 0.0;
    flip_vertical_prob = 0.0;
  }

  std::shared_ptr<TensorToTensorTransform> normalize_transform(new Normalize(torch::ArrayRef<float>(rcnn::config::GetCFG<std::vector<float>>({"INPUT", "PIXEL_MEAN"})),
                                            torch::ArrayRef<float>(rcnn::config::GetCFG<std::vector<float>>({"INPUT", "PIXEL_STD"})),
                                            rcnn::config::GetCFG<bool>({"INPUT", "TO_BGR255"})));
  
  return Compose(
    std::vector<std::shared_ptr<MatToMatTransform>>{
      std::make_shared<Resize>(min_size, max_size),
      std::make_shared<RandomHorizontalFlip>(flip_horizontal_prob),
      std::make_shared<RandomVerticalFlip>(flip_vertical_prob),
    },
    std::vector<std::shared_ptr<TensorToTensorTransform>>{
      normalize_transform
    }
  );
}

}
}