# executables 
list(APPEND include_libraries Compiler Tool)

add_executable(testCompileEngine test_compile_engine.cpp 
                ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)

set_target_properties(testCompileEngine PROPERTIES FOLDER "src")
target_link_libraries(testCompileEngine PUBLIC ${include_libraries})
target_include_directories(testCompileEngine PUBLIC 
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                        )