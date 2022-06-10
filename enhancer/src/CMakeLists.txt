
# gRPC
cmake_minimum_required(VERSION 3.5.1)
# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().
set(CMAKE_PREFIX_PATH "/usr/local/grpc")

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
if(CMAKE_CROSSCOMPILING)
find_program(_PROTOBUF_PROTOC protoc)
else()
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()


# Proto file
get_filename_component(hw_proto "enhancer.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/enhancer.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/enhancer.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/enhancer.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/enhancer.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(SOURCES cuHostMemory.cpp
			cuDeviceMemory.cpp
			cuHostMemoryV2.cpp
			infer_engine.cpp
			encode_engine.cpp
			neural_enhancer.cpp
			neural_enhancer_server.cpp
			remote_neural_enhancer.cpp
			cudaUtility.cu
			vpxenc_api.c
			libvpx_engine.cpp
			$ENV{LIBVPX_LIBRARY_PATH}/tools_common.c 
			$ENV{LIBVPX_LIBRARY_PATH}/y4minput.c 
			$ENV{LIBVPX_LIBRARY_PATH}/ivfenc.c 
			$ENV{LIBVPX_LIBRARY_PATH}/video_writer.c 
			$ENV{LIBVPX_LIBRARY_PATH}/vpxstats.c 
			$ENV{LIBVPX_LIBRARY_PATH}/rate_hist.c 
			$ENV{LIBVPX_LIBRARY_PATH}/warnings.c 
			$ENV{LIBVPX_LIBRARY_PATH}/webmenc.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_common.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_any.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_gcc.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_common.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_any.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_gcc.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/planar_functions.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/cpu_id.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/common/hdr_util.cc 
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxer.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvwriter.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvparser.cc
			$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvreader.cc  
			)

add_library(Enhancer 
  ${hw_grpc_srcs}
  ${hw_proto_srcs}
  ${SOURCES}
)
target_link_libraries(Enhancer PUBLIC
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  z
)

set_target_properties(Enhancer PROPERTIES FOLDER "src")
message("${PROJECT_SOURCE_DIR}/include")

# target_link_libraries(Enhancer PUBLIC  ${include_libraries}
#                     "${PROJECT_SOURCE_DIR}/library/kakadu/libkdu_a82R.so" 
#                     )

target_include_directories(Enhancer PUBLIC
							"${PROJECT_SOURCE_DIR}/library/kakadu" 
							"${PROJECT_SOURCE_DIR}/enhancer/include"
							"${PROJECT_SOURCE_DIR}/compiler/include"
							"${PROJECT_SOURCE_DIR}/controller/include"
							"${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common"
							"$ENV{IPP_LIBRARY_PATH}/include"
							)


