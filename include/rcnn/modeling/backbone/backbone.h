#pragma once
#include "resnet.h"
#include "fpn.h"


namespace rcnn{
namespace modeling{

torch::nn::Sequential BuildResnetBackbone();
torch::nn::Sequential BuildResnetFPNBackbone();
torch::nn::Sequential BuildBackbone();

}
}