set(SOURCES "cudaUtility.cu")

add_library(Common ${SOURCES})
set_target_properties(Common PROPERTIES FOLDER "src")

message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Common PUBLIC
							"${PROJECT_SOURCE_DIR}/NeuralEnhancer/include")