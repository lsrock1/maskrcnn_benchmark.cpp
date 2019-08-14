# maskrcnn_benchmark.cpp
faster rcnn cpp implementation based on maskrcnn-benchmark

# Codes
All code architecture from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)


# Installation
```
export INSTALL_DIR=$PWD
git clone --recursive https://github.com/lsrock1/maskrcnn_benchmark.cpp.git
cd $INSTALL_DIR

//opencv install
//ABI=0 is really important
cd maskrcnn_benchmark.cpp/lib/opencv
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..
make install

//yaml-cpp install
//ABI=0 is really important
cd $INSTALL_DIR
cd maskrcnn_benchmark.cpp/lib/yaml-cpp
mkdir build
cd build
cmake -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 ..
make install

//Download pytorch cpp
//place libtorch folder in lib directory
//if you are using without cuda and get dyld: Library not loaded: @rpath/libmklml.dylib error
//https://github.com/pytorch/pytorch/issues/14165

cd $INSTALL_DIR
cd maskrcnn_benchmark.cpp
mkdir build
cd build
cmake ..
make

//run inference r50-fpn
run.out ../configs/e2e_faster_rcnn_R_50_FPN_1x.yaml inference

```

# Datasets
Download coco datasets  
make directory under maskrcnn_benchmark.cpp
```
datasets
  - coco
    - train2017
    - val2017
  - annotations
    - instances_val2017.json
    - instances_train2017.json
```

# Results

#### Device:  RTX2080ti 1 GPU, cuda 10, cudnn 7, Ubuntu 16.04
#### [ResNet](https://arxiv.org/abs/1512.03385)
#### [VoVNet](https://arxiv.org/abs/1904.09730)
#### ResNets are 1x and VoVNets are 2x

backbone | type | lr sched | inference total batch | inference time(s/im) | box AP | Speed Improvement
-- | -- | -- | -- | -- | -- | --
R-50-FPN(python) | Fast | 1x | 8 | 0.05989 | 0.368 | 0
R-50-FPN(cpp) | Fast | 1x | 8 | 0.05520 | 0.368 | 7.8%
R-101-FPN(python) | Fast | 1x | 8 | 0.07627 | 0.391 | 0
R-101-FPN(cpp) | Fast | 1x | 8 | 0.07176 | 0.391 | 5.9%
VoV-39(python) | Fast | 2x | 8 | 0.06479 | 0.398 | 0
VoV-39(cpp) | Fast | 2x | 8 | 0.05949 | 0.398 | 8.1%
VoV-57(python) | Fast | 2x | 8 | 0.07224 | 0.409 | 0
VoV-57(cpp) | Fast | 2x | 8 | 0.06713 | 0.409 | 7%

# Warning
### In Progress.  
* It doesn't support training yet.(waiting for next libtorch release)

# TODO
- [ ] concat dataset
- [x] python jit -> cpp model code
- [ ] multi GPU training(code complete but bug exists in libtorch)
- [ ] cmake install
- [ ] clean up code

# Requirements
* Yaml-cpp
* gtest
* libtorch >= 1.2
* rapidjson
* opencv

# MODELS
Download into /models 

name | from | link 
-- | -- | -- 
R-50(backbone only) | python-pretrained | [link](https://www.dropbox.com/s/2q808v0p2j75lfq/resnet50_cpp.pth?dl=0)
R-101(backbone only) | python-pretrained | [link](https://www.dropbox.com/s/h5a51ur3qvrdjh5/resnet101_cpp.pth?dl=0)
R-50-FPN | python-trained | [link](https://www.dropbox.com/s/4uvdc8kaluelzx8/frcn_r50_fpn_cpp.pth?dl=0)
R-101-FPN | python-trained | [link](https://www.dropbox.com/s/sgo3k502kegmcxa/frcn_r101_fpn_cpp.pth?dl=0)
R-50-C4 | python-trained | [link](https://www.dropbox.com/s/zu1yzt9ydlnqin4/frcn_r50_c4_cpp.pth?dl=0)
V-39-FPN | python-trained | [link](https://www.dropbox.com/s/h0rgqyy375m3rhv/frcn_v39_fpn_cpp.pth?dl=0)
V-57-FPN | python-trained | [link](https://www.dropbox.com/s/18alpuwz8ft9d86/frcn_v57_fpn_cpp.pth?dl=0)
