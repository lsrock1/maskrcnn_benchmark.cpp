#include "parallel.h"
#include <iostream>


namespace rcnn{
namespace engine{

#ifdef WITH_CUDA

std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<torch::nn::ModuleHolder<rcnn::modeling::GeneralizedRCNNImpl>>& modules,
    const std::vector<rcnn::structures::ImageList>& inputs,
    const std::vector<std::vector<rcnn::structures::BoxList>>& targets,
    const torch::optional<std::vector<torch::Device>>& devices) {
//   TORCH_CHECK(
//       modules.size() == inputs.size(), "Must have as many inputs as modules");
//   if (devices) {
//     TORCH_CHECK(
//         modules.size() == devices->size(),
//         "Must have as many devices as modules");
//   }

  std::vector<std::map<std::string, torch::Tensor>> outputs(modules.size());
  std::mutex mutex;

  // std::exception_ptr can be passed between threads:
  // > An instance of std::exception_ptr may be passed to another function,
  // > possibly on another thread, where the exception may be rethrown [...].
  // https://en.cppreference.com/w/cpp/error/exception_ptr
  std::exception_ptr exception;
  at::parallel_for(
      /*begin=*/0,
      /*end=*/modules.size(),
      /*grain_size=*/1,
      [&modules, &inputs, &targets, &devices, &outputs, &mutex, &exception](
          int64_t index, int64_t stop) {
        for (; index < stop; ++index) {
          try {
            std::map<std::string, torch::Tensor> loss_map = modules[index]->forward<std::map<std::string, torch::Tensor>>(inputs[index], targets[index]);
            auto to_device = (devices ? (*devices)[index] : inputs[index].get_tensors().device());
            
            //remove TODO
            //torch::Tensor loss = torch::zeros({1}).to(to_device).set_requires_grad(true);
            for(auto i = loss_map.begin(); i != loss_map.end(); ++i){
              if(i == loss_map.begin()){
                loss_map["loss"] = i->second;
              }
              else{
                loss_map["loss"] = loss_map["loss"] + i->second;
              }
            }
            // loss_map["loss"] = loss;
            // output =
            //     output.to(devices ? (*devices)[index] : inputs[index].device());
            std::lock_guard<std::mutex> lock(mutex);
            outputs[index] = loss_map;
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!exception) {
              exception = std::current_exception();
            }
          }
        }
      });

  if (exception) {
    std::rethrow_exception(exception);
  }

  return outputs;
}
#endif

std::pair<torch::Tensor, std::map<std::string, torch::Tensor>> data_parallel(
    rcnn::modeling::GeneralizedRCNN module,
    rcnn::structures::ImageList images, 
    std::vector<rcnn::structures::BoxList> targets,
    torch::optional<std::vector<torch::Device>> devices,
    torch::optional<torch::Device> output_device,
    int64_t dim) {
  if (!devices) {
    const auto device_count = torch::cuda::device_count();

    // TORCH_CHECK(
    //     device_count > 0, "Expected at least one CUDA device to be available");
    devices = std::vector<Device>();
    devices->reserve(device_count);
    for (size_t index = 0; index < device_count; ++index) {
      devices->emplace_back(kCUDA, index);
    }
  }
  if (!output_device) {
    output_device = devices->front();
  }

  if (devices->size() >= 1) {
    torch::Tensor loss = torch::zeros({1}).to(devices->front());
    module->to(devices->front());
    images = images.to(devices->front());
    std::vector<rcnn::structures::BoxList> target_device;
    for(auto& box : targets)
      target_device.push_back(box.To(devices->front()));
    auto loss_map = module->forward<std::map<std::string, torch::Tensor>>(images, target_device);//.to(*output_device);
    for(auto i = loss_map.begin(); i != loss_map.end(); ++i){
      loss += i->second;
    }
    loss_map["loss"] = loss;
    return std::make_pair(loss, loss_map);
  }

#ifdef WITH_CUDA
  torch::autograd::Scatter scatter(*devices, /*chunk_sizes=*/torch::nullopt, dim);
  //handle input image_list
  torch::Tensor input = images.get_tensors();
  //because of current bug, set requires true is necessary
  //Remove next release TODO
  input.set_requires_grad(true);
  auto scattered_tensors = fmap<torch::Tensor>(scatter.apply({std::move(input)}));
  std::vector<rcnn::structures::ImageList> scattered_inputs;
  int tensor_index = 0;
  for(auto& tensor : scattered_tensors){
    std::vector<std::pair<int64_t, int64_t>> slice;
    for(int i = 0; i < tensor.size(0); ++i)
      slice.push_back(images.get_image_sizes().at(i + tensor_index));
    scattered_inputs.emplace_back(tensor, slice);
    tensor_index += tensor.size(0);
  }
  
  //handle target bounding_box
  std::vector<std::vector<rcnn::structures::BoxList>> scattered_targets;

  int box_index = 0;
  for(auto& scattered_images : scattered_inputs){
    int size = scattered_images.get_tensors().size(0);
    std::vector<rcnn::structures::BoxList> slice;
    slice.reserve(size);
    for(size_t index = 0; index < size; ++index)
      slice.push_back(targets.at(box_index + index).To(scattered_images.get_tensors().device()));
    scattered_targets.push_back(slice);
    box_index += size;
  }

  auto replicas = torch::nn::parallel::replicate<rcnn::modeling::GeneralizedRCNNImpl>(module, *devices);
  std::vector<std::map<std::string, torch::Tensor>> outputs = parallel_apply(replicas, scattered_inputs, scattered_targets, *devices);
  std::vector<torch::Tensor> total_loss;
  total_loss.reserve(outputs.size());
  for(auto& loss_map : outputs)
    total_loss.push_back(loss_map["loss"].unsqueeze(0));
  
  //to run this
  //this bug must be fixed
  //https://github.com/pytorch/pytorch/pull/20286
  //waiting for release this version...
  std::cout << "end loss cal\n";
  return std::make_pair(torch::autograd::Gather(*output_device, dim)
      .apply(fmap<torch::autograd::Variable>(std::move(total_loss)))
      .front(), outputs[0]);
#else
  AT_ERROR("data_parallel not supported without CUDA");
  return std::make_pair(torch::Tensor(), std::map<std::string, torch::Tensor>{});
#endif
}

}
}