# list(APPEND include_paths ")

# library: tool
set(SOURCES "tool_common.cpp"
    ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp
    $ENV{LIBVPX_LIBRARY_PATH}/webmdec.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/common/hdr_util.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxer.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvwriter.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvparser.cc
    $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvreader.cc 
)
add_library(Tool ${SOURCES})
set_target_properties(Tool PROPERTIES FOLDER "src")
message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Tool PUBLIC
					        "${PROJECT_SOURCE_DIR}/tool/include"
                            "$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm"
                            "${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )


# executables 
list(APPEND include_libraries Compiler Tool)

add_executable(ONNXTOPLAN onnx_to_plan.cpp 
                ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)

set_target_properties(ONNXTOPLAN PROPERTIES FOLDER "src")
target_link_libraries(ONNXTOPLAN PUBLIC ${include_libraries})
target_include_directories(ONNXTOPLAN PUBLIC 
                            "${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common"
                            "${CMAKE_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                        )