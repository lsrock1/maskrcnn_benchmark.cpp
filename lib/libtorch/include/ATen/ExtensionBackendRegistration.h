#pragma once
#include <ATen/Backend.h>
#include <ATen/MSNPUType.h>
#include <ATen/XLAType.h>

namespace at {

template <typename FnPtr>
inline void register_extension_backend_op(
    Backend backend,
    const char * schema,
    FnPtr fn) {
      switch (backend) {
        case Backend::MSNPU:
            MSNPUTypeDispatch::register_function(schema, fn);
            break;
        case Backend::XLA:
            XLATypeDispatch::register_function(schema, fn);
            break;
        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
}

} // namespace at
