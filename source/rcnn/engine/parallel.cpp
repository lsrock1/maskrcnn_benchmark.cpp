#include "parallel.h"


namespace rcnn{
namespace engine{

#ifdef USE_CUDA
std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<rcnn::modeling::GeneralizedRCNN>& modules,
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
            auto loss_map = modules[index]->forward<std::map<std::string, torch::Tensor>>(inputs[index], targets[index]);
            auto to_device = (devices ? (*devices)[index] : inputs[index].get_tensors().device());
            
            torch::Tensor loss = torch::zeros({1}).to(to_device);
            for(auto i = loss_map.begin(); i != loss_map.end(); ++i)
              loss += (i->second).to(to_device);
            loss_map["loss"] = loss;
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

  if (devices->size() == 1) {
    torch::Tensor loss = torch::zeros({1}).to(devices->front());
    module->to(devices->front());
    images = images.to(devices->front());
    std::vector<rcnn::structures::BoxList> target_device;
    for(auto& box : targets)
      target_device.push_back(box.To(devices->front()));
    auto loss_map = module->forward<std::map<std::string, torch::Tensor>>(images, target_device);//.to(*output_device);
    for(auto i = loss_map.begin(); i != loss_map.end(); ++i)
        loss += i->second;
    loss_map["loss"] = loss;
    return std::make_pair(loss, loss_map);
  }

#ifdef USE_CUDA
  torch::autograd::Scatter scatter(*devices, /*chunk_sizes=*/torch::nullopt, dim);
  //handle input image_list
  torch::Tensor input = images.get_tensors();
  auto scattered_tensors = fmap<torch::Tensor>(scatter.apply({std::move(input)}));
  std::vector<rcnn::structres::ImageList> scattered_inputs;
  for(auto& tensor : scattered_tensors)
    scattered_inputs.implace_back(tensor, images.get_image_sizes());
  
  //handle target bounding_box
  std::vector<std::vector<rcnn::structures::BoxList>> scattered_targets;
  
//   std::vector<torch::Tensor> labels;
//   std::vector<torch::Tensor> bboxes;
//   for(auto& box : targets){
//     labels.push_back(box.GetField("labels"))
//     bboxes.push_back(box.get_bbox());
//   }
//   torch::Tensor bboxes_tensor = torch::cat(bboxes, 0), labels_tensor = torch::cat(labels, 0);
//   input = torch::cat({bboxes_tensor, labels_tensor}, 1);

//   scattered_tensors = fmap<torch::Tensor>(scatter.apply({std::move(input)}));
//   int box_index = 0;
//   for(auto& boxes : scattered_tensors){
//     std::vector<rcnn::structures::BoxList> tmp;
//     for(int i = 0; i < boxes.size(0); ++i){
//       targets[i].set_bbox();
//       targets[i].AddField();

//     }
//   }

  int box_index = 0;
  for(auto& images : scattered_inputs){
    int size = images.get_tensors().size(0);
    std::vector<rcnn::structures::BoxList> slice{targets.begin() + box_index, targets.begin() + box_index + size};
    for(auto& i : slice)
      i = i.To(image.get_tensors().device());
    scattered_targets.push_back(slice);
    box_index += size;
  }

  auto replicas = replicate(module, *devices);
  auto outputs = parallel_apply(replicas, scattered_inputs, scattered_targets, *devices);
  std::vector<torch::Tensor> total_loss(outputs.size());
  for(auto& loss_map : outputs)
    total_loss.push_back(loss_map["loss"]);
  {
    torch::NoGradGuard guard;
    for(int start = 1; start < outputs.size(); ++start){
      for(auto i = outputs[start].begin(); i != outputs[start].end(); ++i)
        outputs[0][i->first] += i->second;
    }
  }
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