

list(APPEND include_paths "${PROJECT_SOURCE_DIR}/NeuralEnhancer/include" "${PROJECT_SOURCE_DIR}/NeuralEnhancer/test/Common")
list(APPEND include_libraries Memory Model)

add_executable(testBaseModel testBaseModel.cpp)
set_target_properties(testBaseModel PROPERTIES FOLDER "test")
target_link_libraries(testBaseModel PUBLIC ${include_libraries})
target_include_directories(testBaseModel PUBLIC ${include_paths})

add_executable(testBuild testBuild.cpp)
set_target_properties(testBuild PROPERTIES FOLDER "test")
target_link_libraries(testBuild PUBLIC ${include_libraries})
target_include_directories(testBuild PUBLIC ${include_paths})

add_executable(testStream testStream.cpp)
set_target_properties(testStream PROPERTIES FOLDER "test")
target_link_libraries(testStream PUBLIC ${include_libraries})
target_include_directories(testStream PUBLIC ${include_paths})

add_executable(testLoad testLoad.cpp)
set_target_properties(testLoad PROPERTIES FOLDER "test")
target_link_libraries(testLoad PUBLIC ${include_libraries})
target_include_directories(testLoad PUBLIC ${include_paths})

add_executable(testInfer testInfer.cpp)
set_target_properties(testInfer PROPERTIES FOLDER "test")
target_link_libraries(testInfer PUBLIC ${include_libraries})
target_include_directories(testInfer PUBLIC ${include_paths})