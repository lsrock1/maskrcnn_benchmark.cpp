#include "rpn/anchor_generator.h"
#include <cassert>

#include <defaults.h>


namespace rcnn{
namespace modeling{
//anchor utils
//ratio_enum in original
torch::Tensor RepeatAnchorRatios(torch::Tensor& anchor, torch::Tensor& ratios){
  torch::Tensor w, h, x, y;
  std::tie(w, h, x, y) = WidthHeightCoordXY(anchor);
  //ex) 32 * 32
  torch::Tensor size = w * h;
  //ex) (32 * 32 * 2, 32 * 32 * 1, 32 * 32 * 2)
  torch::Tensor size_ratios = size / ratios;
  torch::Tensor ws = torch::round(torch::sqrt(size_ratios));
  torch::Tensor hs = torch::round(ws * ratios);
  return MakeAnchors(ws, hs, x, y);
}

std::tuple<Width, Height, CoordCenterX, CoordCenterY> WidthHeightCoordXY(torch::Tensor& anchor){
  Width w = anchor[2] - anchor[0] + 1;
  Height h = anchor[3] - anchor[1] + 1;
  CoordCenterX x = anchor[0] + 0.5 * (w - 1);
  CoordCenterY y = anchor[1] + 0.5 * (h - 1);
  return std::make_tuple(w, h, x, y);
}

torch::Tensor MakeAnchors(Width& ws, Height& hs, CoordCenterX& x, CoordCenterY& y){
  ws.unsqueeze_(1);
  hs.unsqueeze_(1);
  // (ratios, 4)
  // 4: {xmin, ymin, xmax, ymax}
  torch::Tensor anchors = torch::cat(
    {
      x - 0.5 * (ws - 1),
      y - 0.5 * (hs - 1),
      x + 0.5 * (ws - 1),
      y + 0.5 * (hs - 1)
    },
    /*dim=*/1
  );
  return anchors;
}

torch::Tensor RepeatAnchorScales(torch::Tensor anchor, torch::Tensor scales){
  torch::Tensor w, h, x, y;
  std::tie(w, h, x, y) = WidthHeightCoordXY(anchor);
  torch::Tensor ws = w * scales, hs = h * scales;
  return MakeAnchors(ws, hs, x, y);
}
  

torch::Tensor GenerateAnchors(int64_t base_size, std::vector<int64_t> anchor_sizes, std::vector<float> aspect_ratios){
  //ex) (0,5, 1, 2)
  torch::Tensor aspect_ratios_tensor = torch::tensor(aspect_ratios).to(torch::kF32);
  //ex) (32, 64, 128, 256, 512) / 16
  torch::Tensor anchor_sizes_tensor = torch::tensor(anchor_sizes).to(torch::kF32) / base_size;
  //(0, 0, base_size-1, base_size-1)
  //base anchor: (0, 0) (base_size-1, base_size-1)
  torch::Tensor anchor = torch::tensor({0., 0., (float) (base_size-1), (float) (base_size-1)}).to(torch::kF32);
  torch::Tensor anchors = RepeatAnchorRatios(anchor, aspect_ratios_tensor);
  std::vector<torch::Tensor> repeated_scale_anchors;
  repeated_scale_anchors.reserve(anchors.size(0));
  for(auto i = 0; i < anchors.size(0); ++i){
    repeated_scale_anchors.push_back(std::move(RepeatAnchorScales(anchors[i], anchor_sizes_tensor)));
  }
  
  return torch::cat(repeated_scale_anchors, /*dim=*/0);
}

  int BufferListsImpl::size(){
    return buffers().size();
  }

  void BufferListsImpl::extend(std::vector<torch::Tensor> buffers){
    int offset = size();
    for(auto& buffer : buffers){
      register_buffer(std::to_string(offset++), buffer);
    }
  }

  torch::Tensor BufferListsImpl::operator[](const int index){
    return named_buffers()[std::to_string(index)];
  }

  AnchorGeneratorImpl::AnchorGeneratorImpl(std::vector<int64_t> sizes, std::vector<float> aspect_ratios, std::vector<int64_t> anchor_strides, int straddle_thresh)
      :strides_(anchor_strides),
       straddle_thresh_(straddle_thresh),
       cell_anchors_(register_module("anchors", BufferLists())){
    std::vector<torch::Tensor> cell_anchors;
    if(anchor_strides.size() == 1){
      int64_t anchor_stride = anchor_strides[0];
      cell_anchors.push_back(std::move(GenerateAnchors(anchor_stride, sizes, aspect_ratios).toType(torch::kFloat32)));
    }
    else{
      cell_anchors.reserve(sizes.size());
      assert(anchor_strides.size() == sizes.size());
      for(int i = 0; i < sizes.size(); ++i){
        cell_anchors.push_back(std::move(GenerateAnchors(anchor_strides[i], std::vector<int64_t> {sizes[i]}, aspect_ratios).toType(torch::kFloat32)));
      }
    }
    cell_anchors_->extend(cell_anchors);
  }

  std::vector<torch::Tensor> AnchorGeneratorImpl::GridAnchors(std::vector<std::pair<int64_t, int64_t>>& grid_sizes){
    std::vector<torch::Tensor> anchors;
    anchors.reserve(grid_sizes.size());
    int64_t stride, grid_height, grid_width;
    torch::Tensor base_anchors, shifts_x, shifts_y, shift_x, shift_y, shifts;
    for(auto i = 0; i < grid_sizes.size(); ++i){
      grid_height = std::get<0>(grid_sizes[i]);
      grid_width = std::get<1>(grid_sizes[i]);
      stride = strides_[i];
      base_anchors = (*cell_anchors_)[i];
      
      shifts_x = torch::arange(
        /*start=*/0, /*end=*/grid_width * stride, /*step=*/stride, torch::TensorOptions().dtype(torch::kFloat32).device(base_anchors.device())
      );
      shifts_y = torch::arange(
        /*start=*/0, /*end=*/grid_height * stride, /*step=*/stride, torch::TensorOptions().dtype(torch::kFloat32).device(base_anchors.device())
      );
      auto meshxy = torch::meshgrid({shifts_y, shifts_x});
      shift_y = meshxy[0];
      shift_x = meshxy[1];
      shift_x = shift_x.reshape({-1});
      shift_y = shift_y.reshape({-1});
      shifts = torch::stack({shift_x, shift_y, shift_x, shift_y}, /*dim=*/1);

      anchors.push_back(
          std::move((shifts.view({-1, 1, 4}) + base_anchors.view({1, -1, 4})).reshape({-1, 4}))
      );
    }
    return anchors;
  }

  std::vector<int64_t> AnchorGeneratorImpl::NumAnchorsPerLocation(){
    std::vector<int64_t> num_anchors;
    num_anchors.reserve(cell_anchors_->size());
    for(int i = 0; i < cell_anchors_->size(); ++i){
      num_anchors.push_back((*cell_anchors_)[i].size(0));
    }
    return num_anchors;
  }
  
  std::vector<std::vector<rcnn::structures::BoxList>> AnchorGeneratorImpl::forward(rcnn::structures::ImageList& image_list, std::vector<torch::Tensor>& feature_maps){
    std::vector<std::pair<int64_t, int64_t>> grid_sizes;
    grid_sizes.reserve(feature_maps.size());
    std::vector<std::vector<rcnn::structures::BoxList>> anchors;
    for(auto i = 0; i < feature_maps.size(); ++i){
      grid_sizes.push_back(std::make_pair(feature_maps[i].size(2), feature_maps[i].size(3)));
    }
    std::vector<torch::Tensor> anchors_over_all_feature_maps = GridAnchors(grid_sizes);
    auto image_sizes = image_list.get_image_sizes();
    anchors.reserve(image_sizes.size());
    std::vector<rcnn::structures::BoxList> anchors_in_image;

    for(auto& image_size: image_sizes){
      anchors_in_image.clear();
      anchors_in_image.reserve(anchors_over_all_feature_maps.size());
      for(auto& anchors_per_feature_map: anchors_over_all_feature_maps){
        rcnn::structures::BoxList boxlist(anchors_per_feature_map, std::make_pair(std::get<1>(image_size), std::get<0>(image_size)), /*mode=*/"xyxy");
        AddVisibilityTo(boxlist);
        anchors_in_image.push_back(std::move(boxlist));
      }
      anchors.push_back(std::move(anchors_in_image));
    }
    return anchors;
  }

  void AnchorGeneratorImpl::AddVisibilityTo(rcnn::structures::BoxList& boxlist){
    int64_t image_width = std::get<0>(boxlist.get_size());
    int64_t image_height = std::get<1>(boxlist.get_size());
    torch::Tensor anchors = boxlist.get_bbox();
    torch::Tensor inds_inside;
    if(straddle_thresh_ >= 0){
      inds_inside = (anchors.select(/*dim=*/1, /*index=*/0) >= -straddle_thresh_)
      .__and__(anchors.select(/*dim=*/1, /*index=*/1) >= -straddle_thresh_)
      .__and__(anchors.select(/*dim=*/1, /*index=*/2) >= (image_width + straddle_thresh_))
      .__and__(anchors.select(/*dim=*/1, /*index=*/3) >= (image_height + straddle_thresh_));
    }
    else{
      inds_inside = torch::ones(anchors.size(0), torch::TensorOptions().dtype(torch::kInt8).device(anchors.device()));
    }
    boxlist.AddField("visibility", inds_inside);
  }

  AnchorGenerator MakeAnchorGenerator(){
    std::vector<int64_t> anchor_sizes = rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_SIZES"});
    std::vector<float> aspect_ratios = rcnn::config::GetCFG<std::vector<float>>({"MODEL", "RPN", "ASPECT_RATIOS"});
    std::vector<int64_t> anchor_stride = rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_STRIDE"});
    int straddle_thresh = rcnn::config::GetCFG<int>({"MODEL", "RPN", "STRADDLE_THRESH"});
    if(rcnn::config::GetCFG<bool>({"MODEL", "RPN", "USE_FPN"}))
      assert(anchor_stride.size() == anchor_sizes.size());
    else
      assert(anchor_stride.size() == 1);
    return AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh);
  }
}
}//rcnn