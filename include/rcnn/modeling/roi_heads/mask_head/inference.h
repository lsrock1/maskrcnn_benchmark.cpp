#pragma once
#include <torch/torch.h>
#include "conv2d.h"
#include "bounding_box.h"


namespace rcnn{
namespace modeling{

torch::Tensor ExpandBoxes(torch::Tensor& boxes, float scale);
std::pair<torch::Tensor, float> ExpandMasks(torch::Tensor mask, int padding);
torch::Tensor PasteMaskInImage(torch::Tensor mask, torch::Tensor box, int64_t im_h, int64_t im_w, float threshold = 0.5, int padding = 1);

class Masker{

public:
  Masker(float threshold = 0.5, int padding = 1);
  Masker(const Masker& other);
  torch::Tensor ForwardSingleImage(torch::Tensor& masks, rcnn::structures::BoxList& boxes);
  std::vector<torch::Tensor> operator()(std::vector<torch::Tensor>& masks, std::vector<rcnn::structures::BoxList>& boxes);

private:
  float threshold_;
  int padding_;
};

class MaskPostProcessorImpl : torch::nn::Module{

public:
  MaskPostProcessorImpl(Masker* masker);
  MaskPostProcessorImpl();
  ~MaskPostProcessorImpl();
  MaskPostProcessorImpl(const MaskPostProcessorImpl& other);
  MaskPostProcessorImpl(MaskPostProcessorImpl&& other);
  MaskPostProcessorImpl& operator=(const MaskPostProcessorImpl& other);
  MaskPostProcessorImpl& operator=(MaskPostProcessorImpl&& other);
  std::vector<rcnn::structures::BoxList> forward(torch::Tensor& x, std::vector<rcnn::structures::BoxList>& boxes);

private:
  Masker* masker_{nullptr};
};

TORCH_MODULE(MaskPostProcessor);

class MaskPostProcessorCOCOFormatImpl : MaskPostProcessorImpl{

public:
  MaskPostProcessorCOCOFormatImpl(Masker* masker): MaskPostProcessorImpl(masker){};
  std::vector<rcnn::structures::BoxList> forward(torch::Tensor& x, std::vector<rcnn::structures::BoxList>& boxes);
};

TORCH_MODULE(MaskPostProcessorCOCOFormat);

MaskPostProcessor MakeRoiMaskPostProcessor();

}//modeling
}//rcnn