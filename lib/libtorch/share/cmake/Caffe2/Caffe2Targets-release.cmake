#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "c10_cuda" for configuration "Release"
set_property(TARGET c10_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10_cuda PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libc10_cuda.so"
  IMPORTED_SONAME_RELEASE "libc10_cuda.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS c10_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_c10_cuda "${_IMPORT_PREFIX}/lib/libc10_cuda.so" )

# Import target "c10" for configuration "Release"
set_property(TARGET c10 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libc10.so"
  IMPORTED_SONAME_RELEASE "libc10.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS c10 )
list(APPEND _IMPORT_CHECK_FILES_FOR_c10 "${_IMPORT_PREFIX}/lib/libc10.so" )

# Import target "caffe2" for configuration "Release"
set_property(TARGET caffe2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe2.so"
  IMPORTED_SONAME_RELEASE "libcaffe2.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe2 "${_IMPORT_PREFIX}/lib/libcaffe2.so" )

# Import target "caffe2_gpu" for configuration "Release"
set_property(TARGET caffe2_gpu APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe2_gpu PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe2_gpu.so"
  IMPORTED_SONAME_RELEASE "libcaffe2_gpu.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe2_gpu )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe2_gpu "${_IMPORT_PREFIX}/lib/libcaffe2_gpu.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
