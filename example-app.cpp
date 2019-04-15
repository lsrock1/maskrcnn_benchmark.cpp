#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include <layers/batch_norm.h>
#include <layers/misc.h>

using namespace std;

int main() {
//   auto bn = FrozenBatchNorm2d(2);
  //Declare 3 dimension tensor with batch dimension
  auto c = eConv2d(torch::nn::Conv2dOptions(3, 3, 3)
                   .stride(1)
                   .padding(1)
                   .with_bias(false));
  auto t = torch::zeros({1, 3, 5, 5});
  cout << c(t) << endl;
  auto m = torch::zeros({0, 3, 5, 5});
  cout << c(m) << endl;
  // cout << c << endl;
  // cout << (tensor2 < 2) << endl;
  // at::Tensor tensor = at::zeros({2, 3});
  // std::cout << tensor << std::endl;
  
  //반환된 리스트의 이름은 포인터이므로 * 으로 첫 번째 값에 접근한다
  // std::cout << *tensor.data<float>() << '\n';
}