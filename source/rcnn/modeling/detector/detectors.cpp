#include "detector/detectors.h"
#include <cassert>


namespace rcnn{
namespace modeling{

GeneralizedRCNN BuildDetectionModel(){
  auto arch = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "META_ARCHITECTURE"});
  std::string arch_name(arch.get());
  if(arch_name.compare("GeneralizedRCNN"))
    return GeneralizedRCNN();

  assert(false);
}

}
}