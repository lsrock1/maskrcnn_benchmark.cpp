#include "detector/detectors.h"
#include <cassert>
#include <iostream>


namespace rcnn{
namespace modeling{

GeneralizedRCNN BuildDetectionModel(){
  std::cout << "start\n";
  auto arch = rcnn::config::GetCFG<std::string>({"MODEL", "META_ARCHITECTURE"});
  std::cout << arch << "\n";
  if(arch.compare("GeneralizedRCNN") == 0)
    return GeneralizedRCNN();

  assert(false);
}

}
}