# CMakeList.txt : CMake project for NeuralEnhancer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Project name
project("NeuralEnhancer")

# Compiler setting (CXX, CUDA)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-deprecated-declarations>")

add_compile_definitions(NDEBUG)
add_compile_definitions(D_LARGEFILE64_SOURCE)
add_compile_definitions(KDU_X86_INTRINSICS)
add_compile_definitions(_FILE_OFFSET_BITS=64)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)
# find_package(CUDA  REQUIRED)

# C++ setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CMake setting
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/library/intel-ipp/lib/intel64/")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Libraries
include_directories(/usr/local/cuda/include)
# include_directories(/usr/local/cuda/targets/x86_64-linux/include)
include_directories($ENV{LIBVPX_LIBRARY_PATH})
include_directories($ENV{LIBVPX_LIBRARY_PATH}/vpx)
include_directories($ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/include)
include_directories($ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/include/libyuv)
include_directories($ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm)
include_directories(${CMAKE_SOURCE_DIR}/library/date/include)
include_directories(${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common)
# include_directories(${CMAKE_SOURCE_DIR}/library/kakadu)
link_directories(/usr/local/cuda/lib64)
link_directories($ENV{LIBVPX_LIBRARY_PATH})
link_directories(/usr/local/cuda/lib64/stubs)
link_directories($ENV{IPP_LIBRARY_PATH}/lib/intel64)
link_libraries(nvinfer nvonnxparser nvidia-ml cudart pthread vpx ippcore ippcc ippvm ipps ippi turbojpeg)

# Build executables (tool)
add_subdirectory(tool/src)

# Build librarires (src)
add_subdirectory(compiler/src)
add_subdirectory(enhancer/src)
add_subdirectory(controller/src)

# Build executables (test)
add_subdirectory(compiler/test)
add_subdirectory(enhancer/test)
add_subdirectory(controller/test)

# Build executables (eval)
add_subdirectory(eval)