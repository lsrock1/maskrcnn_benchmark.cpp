#pragma once
#include "transforms/transforms.h"


namespace rcnn{
namespace data{

Compose BuildTransforms(bool is_train=true);

}
}