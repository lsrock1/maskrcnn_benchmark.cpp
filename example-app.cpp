#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include <layers/batch_norm.h>
#include <typeinfo>
#include <layers/misc.h>

using namespace std;

int main() {
//   auto bn = FrozenBatchNorm2d(2);
  //Declare 3 dimension tensor with batch dimension
  auto c = Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
                   .stride(1)
                   .padding(1)
                   .with_bias(false));
//   cout << c(t) << endl;
  auto m = torch::zeros({0, 3, 5, 5}, torch::TensorOptions().requires_grad(true));
  auto result = c(m);
  auto b = torch::tensor({1, 1, 1, 2, 2, 2, 3, 3, 3}).view({3, 3});//.split(1, -1);
  auto index = torch::tensor({1, 1, 0});
  cout << b[index] << endl;
//   cout << torch::cat(make_tuple(b.at(0), b.at(1), b.at(2)), 1) << endl;
  

  // cout << c << endl;
  // cout << (tensor2 < 2) << endl;
  // at::Tensor tensor = at::zeros({2, 3});
  // std::cout << tensor << std::endl;
  
  //반환된 리스트의 이름은 포인터이므로 * 으로 첫 번째 값에 접근한다
  // std::cout << *tensor.data<float>() << '\n';
}