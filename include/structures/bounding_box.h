#pragma once
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <iostream>

class BoxList{
    static const int FLIP_LEFT_RIGHT = 0;
    static const int FLIP_TOP_BOTTOM = 1;

    public:
        BoxList(torch::Tensor bbox, pair<int64_t, int64_t> image_size, string mode="xyxy");
        void add_field(const string field_name, torch::Tensor field_data);
        torch::Tensor get_field(const string field_name);
        bool has_field(const string field_name);
        vector<string> fields();
        void convert(const string mode);
        BoxList resize(const pair<int64_t, int64_t> size);
        BoxList transpose(const int flip_type);
        BoxList crop(const tuple<int64_t, int64_t, int64_t, int64_t> box);
        BoxList to(const torch::Device device);
        int64_t length();
        BoxList& operator[](const torch::Tensor item);
        ostream& operator << ();
        BoxList& clip_to_image(const bool remove_empty=true);
        int64_t area();
        BoxList copy_with_fields(vector<string> fields, const bool skip_missing);
        map<string, torch::Tensor> extra_fields;

    private:
        torch::Device device;
        torch::Tensor bbox;
        pair<int64_t, int64_t> size;
        string mode;
        void _copy_extra_fields(BoxList box);
        tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> _split_into_xyxy();
};