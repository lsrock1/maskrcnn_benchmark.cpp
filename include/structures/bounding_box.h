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
        void Convert(const string mode);
        void Resize(const pair<int64_t, int64_t> size);
        void Transpose(const int flip_type);
        void Crop(const tuple<int64_t, int64_t, int64_t, int64_t> box);
        void To(const torch::Device device);
        //TODO
        // void Convert(const string mode, BoxList& dst_bbox);
        // void Resize(const pair<int64_t, int64_t> size, BoxList& dst_bbox);
        // void Transpose(const int flip_type, BoxList& dst_bbox);
        // void Crop(const tuple<int64_t, int64_t, int64_t, int64_t> box, BoxList& dst_bbox);
        // void To(const torch::Device device, BoxList& dst_bbox);
        int64_t Length() const;
        BoxList& operator[](torch::Tensor item);
        BoxList& operator[](const int64_t index);
        
        void ClipToImage(const bool remove_empty=true);
        torch::Tensor Area();
        void CopyWithFields(vector<string> fields, BoxList dst_bbox, const bool skip_missing=false);
        map<string, torch::Tensor> get_extra_fields() const;
        pair<int64_t, int64_t> get_size() const;
        torch::Device get_device() const;
        torch::Tensor get_bbox() const;
        string get_mode() const;
        void set_size(pair<int64_t, int64_t> size);
        void set_extra_fields(map<string, torch::Tensor> fields);
        void set_device(torch::Device device);
        void set_bbox(torch::Tensor bbox);
        void set_mode(string mode);

    private:
        map<string, torch::Tensor> extra_fields_;
        torch::Device device_;
        torch::Tensor bbox_;
        pair<int64_t, int64_t> size_;
        string mode_;
        void CopyExtraFields(BoxList box);
        tuple<XMin, YMin, XMax, YMax> SplitIntoXYXY();
    friend ostream& operator << (ostream& os, const BoxList& bbox);
};