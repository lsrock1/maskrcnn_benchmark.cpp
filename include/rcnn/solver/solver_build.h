#pragma once
#include "optimizer.h"
#include "lr_scheduler.h"
#include "detector/generalized_rcnn.h"


namespace rcnn{
namespace solver{

ConcatOptimizer MakeOptimizer(rcnn::modeling::GeneralizedRCNN& model);

ConcatScheduler MakeLRScheduler(ConcatOptimizer& optimizer, int last_epoch);

}
}