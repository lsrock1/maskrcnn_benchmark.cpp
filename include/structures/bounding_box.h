#pragma once
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <iostream>

using namespace std;

class BoxList{
    typedef torch::Tensor XMin, YMin, XMax, YMax;

    static const int kFlipLeftRight = 0;
    static const int kFlipTopBottom = 1;

    public:
        BoxList();
        BoxList(torch::Tensor bbox, pair<int64_t, int64_t> image_size, const char* mode="xyxy");
        BoxList(torch::Tensor bbox, pair<int64_t, int64_t> image_size, string mode="xyxy");
        void AddField(const string field_name, torch::Tensor field_data);
        torch::Tensor GetField(const string field_name);
        bool HasField(const string field_name);
        vector<string> Fields();
        BoxList Convert(const string mode);
        BoxList Resize(const pair<int64_t, int64_t> size);
        BoxList Transpose(const int flip_type);
        BoxList Crop(const tuple<int64_t, int64_t, int64_t, int64_t> box);
        BoxList To(const torch::Device device);
        int64_t Length() const;
        BoxList operator[](torch::Tensor item);
        BoxList operator[](const int64_t index);
        
        BoxList ClipToImage(const bool remove_empty=true);
        torch::Tensor Area();
        BoxList CopyWithFields(const vector<string> fields, const bool skip_missing=false);
        map<string, torch::Tensor> get_extra_fields() const;
        pair<int64_t, int64_t> get_size() const;
        torch::Device get_device() const;
        torch::Tensor get_bbox() const;
        string get_mode() const;
        void set_size(const pair<int64_t, int64_t> size);
        void set_extra_fields(const map<string, torch::Tensor> fields);
        void set_bbox(const torch::Tensor bbox);
        void set_mode(const string mode);
        void set_mode(const char* mode);

    private:
        map<string, torch::Tensor> extra_fields_;
        torch::Device device_;
        torch::Tensor bbox_;
        pair<int64_t, int64_t> size_;
        string mode_;
        void CopyExtraFields(const BoxList box);
        tuple<XMin, YMin, XMax, YMax> SplitIntoXYXY();
    friend ostream& operator << (ostream& os, const BoxList& bbox);
};