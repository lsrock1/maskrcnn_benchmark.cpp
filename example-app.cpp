#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

int main() {
  at::Tensor tensor = at::zeros({2, 3});
  std::cout << tensor << std::endl;
  torch::Tensor tensor2 = torch::rand({2, 3});
  std::cout << tensor2 << std::endl;
  tensor = tensor2;
  std::cout << *tensor.data<float>() << '\n';
  // at::Tensor& ate = tensor2.data();
  // std::cout << ate << std::endl;
}