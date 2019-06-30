#pragma once
#include <torch/torch.h>
#include <modeling.h>
#include <solver_build.h>


namespace rcnn{
namespace utils{

class Checkpoint{

public:
  Checkpoint(rcnn::modeling::GeneralizedRCNN& model, 
             rcnn::solver::ConcatOptimizer& optimizer, 
             rcnn::solver::ConcatScheduler& scheduler, 
             std::string save_dir);

  int load(std::string weight_path);
  void save(std::string name, int iteration);
  bool has_checkpoint();
  std::string get_checkpoint_file();
  int load_from_checkpoint();
  void write_checkpoint_file(std::string name);

private:
  rcnn::modeling::GeneralizedRCNN& model_;
  rcnn::solver::ConcatOptimizer& optimizer_;
  rcnn::solver::ConcatScheduler& scheduler_;
  std::string save_dir_;
};

}
}