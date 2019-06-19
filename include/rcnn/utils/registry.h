#pragma once
#include <torch/torch.h>
#include <modeling.h>
#include <string>


namespace rcnn{
namespace registry{

using backbone = rcnn::modeling::Backbone (*) (void);
backbone BACKBONES(std::string conv_body);

//hard code
}
}

