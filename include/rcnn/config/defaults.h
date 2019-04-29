#pragma once
#include <torch/torch.h>

using namespace std;

namespace rcnn{
namespace configs{

class CFG{

  private:
    map<string, string> string_cfg_;
    map<string, bool> bool_cfg_;
    map<string, vector<double>> double_vec_cfg_;
    map<string, vector<string>> string_vec_cfg_;
    map<string, int64_t> int_cfg_;
    map<string, vector<CFG>> childern_;
    void SetCFGFromFile();
    template<typename T> void SetDatum(const char* node_name, T& datum);
};

CFG GetDefaultCFG();

}//configs
}//mrcn