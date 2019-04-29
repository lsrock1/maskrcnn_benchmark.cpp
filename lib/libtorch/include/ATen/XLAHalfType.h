#pragma once
#include <ATen/XLAType.h>

namespace at {

struct CAFFE2_API XLAHalfType : public XLAType {
  explicit XLAHalfType();

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
};

} // namespace at
