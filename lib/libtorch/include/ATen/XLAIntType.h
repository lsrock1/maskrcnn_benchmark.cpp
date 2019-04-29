#pragma once
#include <ATen/XLAType.h>

namespace at {

struct CAFFE2_API XLAIntType : public XLAType {
  explicit XLAIntType();

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
};

} // namespace at
