#pragma once
#include <ATen/XLAType.h>

namespace at {

struct CAFFE2_API XLADoubleType : public XLAType {
  explicit XLADoubleType();

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
};

} // namespace at
