#pragma once
#include <torch/torch.h>
#include <vector>
#include <image_list.h>
#include <bounding_box.h>


namespace rcnn{
namespace modeling{
  using Width = torch::Tensor;
  using Height = torch::Tensor;
  using CoordCenterX = torch::Tensor;
  using CoordCenterY = torch::Tensor;

    //anchor utils
  torch::Tensor RepeatAnchorRatios(torch::Tensor& anchor, torch::Tensor& ratios);
  std::tuple<Width, Height, CoordCenterX, CoordCenterY> WidthHeightCoordXY(torch::Tensor& anchor);
  torch::Tensor MakeAnchors(Width& ws, Height& hs, CoordCenterX& x, CoordCenterY& y);
  torch::Tensor RepeatAnchorScales(torch::Tensor anchor, torch::Tensor scales);


  torch::Tensor GenerateAnchors(int64_t base_size, std::vector<int64_t> anchor_sizes, std::vector<float> aspect_ratios);

  class BufferListsImpl : public torch::nn::Module{
    public:
      BufferListsImpl() = default;
      int size();
      void extend(std::vector<torch::Tensor> buffers);
      torch::Tensor operator[](const int index);
  };

  TORCH_MODULE(BufferLists);

  class AnchorGeneratorImpl : public torch::nn::Module{
    public:
      AnchorGeneratorImpl(std::vector<int64_t> sizes, std::vector<float> aspect_ratios, std::vector<int64_t> anchor_strides, int straddle_thresh=0);
      std::vector<std::vector<rcnn::structures::BoxList>> forward(rcnn::structures::ImageList& image_list, std::vector<torch::Tensor>& feature_maps);
      std::vector<torch::Tensor> GridAnchors(std::vector<std::pair<int64_t, int64_t>>& grid_sizes);
      void AddVisibilityTo(rcnn::structures::BoxList& boxlist);
      std::vector<int64_t> NumAnchorsPerLocation();
      
    private:
      BufferLists cell_anchors_;
      std::vector<int64_t> strides_;
      int straddle_thresh_;
  };

  TORCH_MODULE(AnchorGenerator);

  AnchorGenerator MakeAnchorGenerator();
}
}//rcnn