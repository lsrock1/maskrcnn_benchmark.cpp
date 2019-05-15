#include <anchor_generator.h>


namespace rcnn{
namespace modeling{
  namespace{
    //anchor utils
    //ratio_enum in original
    torch::Tensor RepeatAnchorRatios(torch::Tensor anchor, torch::Tensor ratios){
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

    std::tuple<Width, Height, CoordCenterX, CoordCenterY> WidthHeightCoordXY(torch::Tensor anchor){
      Width w = anchor[2] - anchor[0] + 1;
      Height h = anchor[3] - anchor[1] + 1;
      CoordCenterX x = anchor[0] + 0.5 * (w - 1);
      CoordCenterY y = anchor[1] + 0.5 * (h - 1);
      return std::make_tuple(w, h, x, y);
    }

    torch::Tensor MakeAnchors(Width ws, Height hs, CoordCenterX x, CoordCenterY y){
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
  }

  torch::Tensor GenerateAnchors(int64_t base_size, std::vector<int64_t> anchor_sizes, std::vector<double> aspect_ratios){
    //ex) (0,5, 1, 2)
    torch::Tensor aspect_ratios_tensor = torch::tensor(aspect_ratios).to(torch::kF64);
    //ex) (32, 64, 128, 256, 512) / 16
    torch::Tensor anchor_sizes_tensor = torch::tensor(anchor_sizes).to(torch::kF64) / base_size;
    //(0, 0, base_size-1, base_size-1)
    //base anchor: (0, 0) (base_size-1, base_size-1)
    torch::Tensor anchor = torch::tensor({0., 0., (double) (base_size-1), (double) (base_size-1)}).to(torch::kF64);
    torch::Tensor anchors = RepeatAnchorRatios(anchor, aspect_ratios_tensor);
    std::vector<torch::Tensor> repeated_scale_anchors;
    for(auto i = 0; i < anchors.size(0); ++i){
      repeated_scale_anchors.push_back(RepeatAnchorScales(anchors[i], anchor_sizes_tensor));
    }
    
    return torch::cat(repeated_scale_anchors, /*dim=*/0);
  }

  int BufferLists::size(){
    return buffers().size();
  }

  void BufferLists::extend(std::vector<torch::Tensor> buffers){
    int offset = size();
    for(auto& buffer : buffers){
      register_buffer(std::to_string(offset++), buffer);
    }
  }

  torch::Tensor BufferLists::operator[](const int index){
    return named_buffers()[std::to_string(index)];
  }

  AnchorGeneratorImpl::AnchorGeneratorImpl(std::vector<int64_t> sizes, std::vector<double> aspect_ratios, std::vector<int64_t> anchor_strides, int straddle_thresh)
      :strides_(anchor_strides),
       straddle_thresh_(straddle_thresh){
    std::vector<torch::Tensor> cell_anchors;
    if(anchor_strides.size() == 1){
      int64_t anchor_stride = anchor_strides[0];
      cell_anchors.push_back(GenerateAnchors(anchor_stride, sizes, aspect_ratios).toType(torch::kFloat64));
    }
    else{
      if(anchor_strides.size() != sizes.size())
        throw "FPN should have #anchor_strides == #sizes";
      for(int i = 0; i < sizes.size(); ++i){
        std::vector<int64_t> size{sizes[i]};
        cell_anchors.push_back(GenerateAnchors(anchor_strides[i], size, aspect_ratios).toType(torch::kFloat64));
      }
    }
    cell_anchors_.extend(cell_anchors);
  }

  std::vector<torch::Tensor> AnchorGeneratorImpl::GridAnchors(std::vector<std::pair<int64_t, int64_t>> grid_sizes){
    std::vector<torch::Tensor> anchors;
    int64_t stride, grid_height, grid_width;
    torch::Tensor base_anchors;
    for(auto i = 0; i < grid_sizes.size(); ++i){
      grid_height = std::get<0>(grid_sizes[i]);
      grid_width = std::get<1>(grid_sizes[i]);
      stride = strides_[i];
      base_anchors = cell_anchors_[i];
      
      torch::Tensor shifts_x = torch::arange(
        /*start=*/0, /*end=*/grid_width * stride, /*step=*/stride, torch::TensorOptions().dtype(torch::kFloat64).device(base_anchors.device())
      );
      torch::Tensor shifts_y = torch::arange(
        /*start=*/0, /*end=*/grid_height * stride, /*step=*/stride, torch::TensorOptions().dtype(torch::kFloat64).device(base_anchors.device())
      );
      auto meshxy = torch::meshgrid({shifts_y, shifts_x});
      torch::Tensor shift_y = meshxy[0], shift_x = meshxy[1];
      shift_x = shift_x.reshape({-1});
      shift_y = shift_y.reshape({-1});
      torch::Tensor shifts = torch::stack({shift_x, shift_y, shift_x, shift_y}, /*dim=*/1);

      anchors.push_back(
          (shifts.view({-1, 1, 4}) + base_anchors.view({1, -1, 4})).reshape({-1, 4})
      );
    }
    return anchors;
  }
  
  std::vector<std::vector<rcnn::structures::BoxList>> AnchorGeneratorImpl::forward(rcnn::structures::ImageList image_list, std::deque<torch::Tensor> feature_maps){
    std::vector<std::pair<int64_t, int64_t>> grid_sizes;
    std::vector<std::vector<rcnn::structures::BoxList>> anchors;
    for(auto i = 0; i < feature_maps.size(); ++i){
      grid_sizes.push_back(std::make_pair(feature_maps[i].size(1), feature_maps[i].size(2)));
    }
    std::vector<torch::Tensor> anchors_over_all_feature_maps = GridAnchors(grid_sizes);
    auto image_sizes = image_list.get_image_sizes();
    for(auto& image_size: image_sizes){
      std::vector<rcnn::structures::BoxList> anchors_in_image;
      for(auto& anchors_per_feature_map: anchors_over_all_feature_maps){
        rcnn::structures::BoxList boxlist(anchors_per_feature_map, std::make_pair(std::get<1>(image_size), std::get<0>(image_size)), /*mode=*/"xyxy");
        AddVisibilityTo(boxlist);
        anchors_in_image.push_back(boxlist);
      }
      anchors.push_back(anchors_in_image);
    }
    return anchors;
  }

  void AnchorGeneratorImpl::AddVisibilityTo(rcnn::structures::BoxList& boxlist){
    int64_t image_width = std::get<0>(boxlist.get_size());
    int64_t image_height = std::get<1>(boxlist.get_size());
    torch::Tensor anchors = boxlist.get_bbox();
    torch::Tensor inds_inside;
    if(straddle_thresh_ >= 0){
      inds_inside = (anchors.slice(/*dim=*/1, /*start=*/0, /*end=*/1) >= -straddle_thresh_)
      .__and__(anchors.slice(/*dim=*/1, /*start=*/1, /*end=*/2) >= -straddle_thresh_)
      .__and__(anchors.slice(/*dim=*/1, /*start=*/2, /*end=*/3) >= (image_width + straddle_thresh_))
      .__and__(anchors.slice(/*dim=*/1, /*start=*/3, /*end=*/4) >= (image_height + straddle_thresh_));
    }
    else{
      inds_inside = torch::ones(anchors.size(0), torch::TensorOptions().dtype(torch::kInt8).device(anchors.device()));
    }
    boxlist.AddField("visibility", inds_inside);
  }
}
}//rcnn