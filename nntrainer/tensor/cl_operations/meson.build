cl_op_sources = [
  'blas_kernels.cpp',
  'blas_kernel_interface.cpp',
]

cl_op_headers = [
  'blas_kernel_interface.h',
]

if get_option('enable-fp16')
  cl_op_sources += 'blas_kernels_fp16.cpp'
endif

foreach s : cl_op_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : cl_op_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach
