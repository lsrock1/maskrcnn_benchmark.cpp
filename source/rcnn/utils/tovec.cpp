#include "tovec.h"
#include <algorithm>


namespace rcnn{
namespace config{

std::vector<std::string> tovec(const char* name){
  std::vector<std::string> splitted;
  std::string svalue(name);
  if(svalue.find("(") != std::string::npos && svalue.find(")") != std::string::npos && std::count(svalue.begin(), svalue.end(), ',') > 0){
    
    size_t pos = 0;
    //remove white spaces
    std::string::iterator end_pos = std::remove(svalue.begin(), svalue.end(), ' ');
    svalue.erase(end_pos, svalue.end());
    //remove ( and )
    svalue = svalue.substr(1, svalue.size()-2);
    std::string token;
    while ((pos = svalue.find(",")) != std::string::npos) {
      token = svalue.substr(0, pos);
      splitted.push_back(token);
      svalue.erase(0, pos + 1);
    }
    if(svalue.size() > 1){
      splitted.push_back(svalue);
    }
  }
  return splitted;
}

}
}