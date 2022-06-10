list(APPEND include_paths "${PROJECT_SOURCE_DIR}/NeuralEnhancer/include" "${PROJECT_SOURCE_DIR}/NeuralEnhancer/test/Common")
list(APPEND include_libraries Memory)

add_executable(testHostCorrectness testHostCorrectness.cpp)
set_target_properties(testHostCorrectness PROPERTIES FOLDER "test")
target_link_libraries(testHostCorrectness PUBLIC ${include_libraries})
target_include_directories(testHostCorrectness PUBLIC ${include_paths})

add_executable(testMemory testMemory.cpp)
set_target_properties(testMemory PROPERTIES FOLDER "test")
target_link_libraries(testMemory PUBLIC ${include_libraries})
target_include_directories(testMemory PUBLIC ${include_paths})