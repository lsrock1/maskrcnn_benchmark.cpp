# maskrcnn_benchmark.cpp
faster rcnn cpp implementation based on maskrcnn-benchmark

# Codes
All code architecture from [mask](https://github.com/facebookresearch/maskrcnn-benchmark)

# Install
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

//fix libtorch bug
//https://github.com/pytorch/pytorch/pull/21134
//in maskrcnn_benchmark.cpp/lib/libtorch/include/torch/csrc/autograd/VariableTypeUtils.h
//insert #pramga once into the first line

cd $INSTALL_DIR
cd maskrcnn_benchmark.cpp
mkdir build
cd build
cmake ..
make

//run inference r50-fpn
./tmp.out
```
# Results
R-50-FPN(cpp) weight if from maskrcnn-benchmark pretrained model  

#### Under RTX2080ti 1gpu, cuda 10, cudnn 7  
  
backbone | type | lr sched | im / gpu | inference time(s/im) | box AP
-- | -- | -- | -- | -- | --
R-50-FPN(python) | Fast | 1x | 1 | 0.05989 | 0.368
R-50-FPN(cpp) | Fast | 1x | 1 | 0.05296 | 0.368

# Warning
### In Progress.  
It doesn't support training yet.(only support single gpu training but not tested).  
It only tested in pre-trained model from python maskrcnn-benchmark.  
It doesn't import evaluation process  

# TODO
- [ ] Other model test
- [ ] python jit -> cpp model code
- [ ] Multi GPU training
- [ ] installable
- [ ] clean up code

# Requirements
Yaml-cpp
gtest
libtorch
rapidjson
opencv
