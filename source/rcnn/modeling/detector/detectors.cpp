#include "detector/detectors.h"
#include <cassert>


namespace rcnn{
namespace modeling{

GeneralizedRCNN BuildDetectionModel(){
  auto arch = rcnn::config::GetCFG<std::string>({"MODEL", "META_ARCHITECTURE"});
  if(arch.compare("GeneralizedRCNN") == 0)
    return GeneralizedRCNN();

  assert(false);
}

}
}