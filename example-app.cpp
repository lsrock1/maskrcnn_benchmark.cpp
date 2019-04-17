#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include <layers/batch_norm.h>
#include <typeinfo>
#include <layers/misc.h>
#include <structures/bounding_box.h>


using namespace std;

int main() {
//   auto bn = FrozenBatchNorm2d(2);
  //Declare 3 dimension tensor with batch dimension
  auto c = Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
                   .stride(1)
                   .padding(1)
                   .with_bias(false));
//   cout << c(t) << endl;
//   auto m = torch::ones({3, 3}, torch::TensorOptions().requires_grad(true));
//   //auto result = c(m);
//   m[0].clamp_max_(0);
//   cout << m << endl;
//   m.narrow({, })
//   cout << torch::cat(make_tuple(b.at(0), b.at(1), b.at(2)), 1) << endl;
  

  // cout << c << endl;
  // cout << (tensor2 < 2) << endl;
  // at::Tensor tensor = at::zeros({2, 3});
  // std::cout << tensor << std::endl;
  
  //반환된 리스트의 이름은 포인터이므로 * 으로 첫 번째 값에 접근한다
  // std::cout << *tensor.data<float>() << '\n';

  ////
  torch::Tensor box = torch::tensor({1, 1, 4, 4, 10, 10, 50, 50}).reshape({2, 4});
  BoxList bb = BoxList(box, make_pair(100, 120), "xyxy");
  cout << bb;
  ////
  bb.AddField("labels", torch::tensor({1, 1}));
  bb.AddField("scores", torch::tensor({0.4, 0.7}));
  auto scores = bb.GetField("scores");
  cout << bb.GetField("scores") << endl;
  bb.Convert("xywh");
  cout << bb << endl;
  cout << "before resize" << endl;
  cout << bb.get_bbox() << endl;
  bb.Resize(make_pair(200, 260));
  cout << "after resize" << endl;
  cout << bb.get_bbox() << endl;
  auto score_mask = scores > 0.5;
  cout << "scores over 0.5 : " <<endl;
  cout << bb<< endl;
  cout << score_mask << endl;
  cout << bb[score_mask] << endl;
  cout << bb.get_bbox() << endl;
  cout << bb[0].get_bbox() << endl;
//   cout << bb;
}