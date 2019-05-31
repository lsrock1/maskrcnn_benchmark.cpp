#pragma once
#include <torch/torch.h>
#include "box_coder.h"
#include "bounding_box.h"


namespace rcnn{
namespace modeling{

class PostProcessor : public torch::nn::Module{
  public:
    PostProcessor(float score_thresh, float nms, int64_t detections_per_img, BoxCoder box_coder);
}

}
}