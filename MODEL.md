# MODEL import from PYTHON

##### This project's purpose is 'train in python, inference in cpp'

## Overall Process

1. Train weight file in maskrcnn-benchmark
2. Using [to_jit.py](./python_utils/to_jit.py) script, convert model's weight file from python to pytorch jit.  
   This script only takes weight except logics(eg. nms..)
3. Using [jit_to_cpp.cpp](./include/rcnn/utils/jit_to_cpp.h), convert pytorch jit weight to libtorch weight.  
   If you are using new type of backbone(not resnet or vovnet), you have to declare and define new backbone mapper.  
   It makes hash map python weight name <-> cpp weight name
4. Saved as /models/new_pth_from_python_cpp.pth
5. Rename and use