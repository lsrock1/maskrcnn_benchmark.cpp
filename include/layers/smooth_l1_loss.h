#pragma once
#include <torch/torch.h>

torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, float beta=1. / 9, bool size_average=true);