#include "solver_build.h"
#include "defaults.h"


namespace rcnn{
namespace solver{

ConcatOptimizer MakeOptimizer(rcnn::modeling::GeneralizedRCNN& model){
  return ConcatOptimizer(model->named_parameters(), 
                        rcnn::config::GetCFG<double>({"SOLVER", "BASE_LR"}),
                        rcnn::config::GetCFG<double>({"SOLVER", "MOMENTUM"}),
                        rcnn::config::GetCFG<double>({"SOLVER", "WEIGHT_DECAY"}));
}

ConcatScheduler MakeLRScheduler(ConcatOptimizer& optimizer){
  auto method = rcnn::config::GetCFG<rcnn::config::CFGS>({"SOLVER", "WARMUP_METHOD"});
  std::string method_name = method.get();
  return ConcatScheduler(optimizer, 
                        rcnn::config::GetCFG<std::vector<int64_t>>({"SOLVER", "STEPS"}),
                        rcnn::config::GetCFG<float>({"SOLVER", "GAMMA"}),
                        rcnn::config::GetCFG<float>({"SOLVER", "WARMUP_FACTOR"}),
                        rcnn::config::GetCFG<int64_t>({"SOLVER", "WARMUP_ITERS"}),
                        method_name);
}

}
}