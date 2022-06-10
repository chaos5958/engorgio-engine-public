set(SOURCES BaseModel.cpp)

add_library(Model ${SOURCES})
set_target_properties(Model PROPERTIES FOLDER "src")
message("${PROJECT_SOURCE_DIR}/include")
target_link_libraries(Model PUBLIC Common Memory)
target_include_directories(Model PUBLIC
							"${PROJECT_SOURCE_DIR}/NeuralEnhancer/include")