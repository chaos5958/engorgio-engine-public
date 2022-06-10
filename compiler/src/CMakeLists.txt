# set(SOURCES "controller.cpp" "decode_engine.cpp" "anchor_selection_engine.cpp")
set(SOURCES "compile_engine.cpp" "compile_common.cpp" "${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp")


add_library(Compiler ${SOURCES})
set_target_properties(Compiler PROPERTIES FOLDER "src")

message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Compiler PUBLIC
							"${PROJECT_SOURCE_DIR}/compiler/include"
							"${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common"
							)