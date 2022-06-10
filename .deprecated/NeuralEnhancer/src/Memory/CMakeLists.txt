set(SOURCES cuHostMemory.cpp
			cuDeviceMemory.cpp)

add_library(Memory ${SOURCES})
set_target_properties(Memory PROPERTIES FOLDER "src")
message("${PROJECT_SOURCE_DIR}/include")
target_include_directories(Memory PUBLIC
							"${PROJECT_SOURCE_DIR}/NeuralEnhancer/include")