set(SOURCES "DecodeEngine.cpp")

add_library(Engine ${SOURCES})
set_target_properties(Engine PROPERTIES FOLDER "src")

message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Engine PUBLIC
							"${PROJECT_SOURCE_DIR}/NeuralEnhancer/include")