#include "checkpoint.h"
#include <fstream>
#include <iostream>
#include <torch/serialize/archive.h>


namespace rcnn{
namespace utils{

Checkpoint::Checkpoint(rcnn::modeling::GeneralizedRCNN& model, 
                    rcnn::solver::ConcatOptimizer& optimizer, 
                    rcnn::solver::ConcatScheduler& scheduler, 
                    std::string save_dir)
                    :model_(model),
                     optimizer_(optimizer),
                     scheduler_(scheduler),
                     save_dir_(save_dir){}

int Checkpoint::load(std::string weight_path){
  //return iteration
  torch::serialize::InputArchive archive;
  if(has_checkpoint()){
    std::string checkpoint_name = get_checkpoint_file();
    archive.load_from(checkpoint_name);
    model_->load(archive);
    optimizer_.load(archive);
    scheduler_.load(archive);
    torch::Tensor iter = torch::zeros({1});
    archive.read("iteration", iter, true);
    return iter.item<int>();
  }
  else{
    //no optimizer scheduler
    archive.load_from(weight_path);
    for(auto& i : model_->named_parameters())
      archive.try_read(i.key(), i.value());
    for(auto& i : model_->named_buffers())
      archive.try_read(i.key(), i.value(), true);
    return 0;
  }
}

void Checkpoint::save(std::string name, int iteration){
  torch::serialize::OutputArchive archive;
  auto iter = torch::tensor({iteration}).to(torch::kI64);
  archive.write("iteration", iter, true);
  model_->save(archive);
  optimizer_.save(archive);
  scheduler_.save(archive);
  archive.save_to(save_dir_ + name);
  write_checkpoint_file(name);
}

bool Checkpoint::has_checkpoint(){
  std::ifstream f(save_dir_ + "/last_checkpoint");
  return f.good();
}

std::string Checkpoint::get_checkpoint_file(){
  std::ifstream inFile(save_dir_ + "/last_checkpoint");
  std::string checkpoint_name;
  std::getline(inFile, checkpoint_name);
  return checkpoint_name;
}

void Checkpoint::write_checkpoint_file(std::string name){
  std::ofstream writeFile(save_dir_ + "/last_checkpoint");
  writeFile << save_dir_ + name;
  writeFile.close();
}


}
}