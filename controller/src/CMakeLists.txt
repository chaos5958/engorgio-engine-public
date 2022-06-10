# set(SOURCES "controller.cpp" "decode_engine.cpp" "anchor_selection_engine.cpp")
set(SOURCES "controller.cpp" "decode_engine.cpp" "control_common.cpp" "anchor_engine.cpp")

add_library(Controller ${SOURCES})
set_target_properties(Controller PROPERTIES FOLDER "src")

message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Controller PUBLIC
							${PROJECT_SOURCE_DIR}/controller/include
							$ENV{IPP_LIBRARY_PATH}/include
							${PROJECT_SOURCE_DIR}/enhancer/include
							)
