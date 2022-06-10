# executables 
list(APPEND include_libraries Controller Compiler Tool Enhancer)

add_executable(benchmark_infer benchmark_infer.cpp
                ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)

set_target_properties(benchmark_infer PROPERTIES FOLDER "src")
target_link_libraries(benchmark_infer PUBLIC ${include_libraries})
target_include_directories(benchmark_infer PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )

add_executable(benchmark_decode benchmark_decode.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)

set_target_properties(benchmark_decode PROPERTIES FOLDER "src")
target_link_libraries(benchmark_decode PUBLIC ${include_libraries})
target_include_directories(benchmark_decode PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )

add_executable(benchmark_encode benchmark_encode.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)

set_target_properties(benchmark_encode PROPERTIES FOLDER "src")
target_link_libraries(benchmark_encode PUBLIC ${include_libraries})
target_include_directories(benchmark_encode PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )

add_executable(benchmark_engorgio_sync benchmark_engorgio_synch.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
            
set_target_properties(benchmark_engorgio_sync PROPERTIES FOLDER "src")
target_link_libraries(benchmark_engorgio_sync PUBLIC ${include_libraries})
target_include_directories(benchmark_engorgio_sync PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            "${CMAKE_CURRENT_BINARY_DIR}/../enhancer/src/"
                            )

add_executable(benchmark_perframe_sync benchmark_perframe_synch.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
            
set_target_properties(benchmark_perframe_sync PROPERTIES FOLDER "src")
target_link_libraries(benchmark_perframe_sync PUBLIC ${include_libraries})
target_include_directories(benchmark_perframe_sync PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                        )

add_executable(benchmark_engorgio_sync_server benchmark_engorgio_sync_server.cpp
                ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
            
set_target_properties(benchmark_engorgio_sync_server PROPERTIES FOLDER "src")
target_link_libraries(benchmark_engorgio_sync_server PUBLIC ${include_libraries})
target_include_directories(benchmark_engorgio_sync_server PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            "${CMAKE_CURRENT_BINARY_DIR}/../enhancer/src/"
                            )
                            

add_executable(benchmark_perframe_async benchmark_perframe_asynch.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
            
set_target_properties(benchmark_perframe_async PROPERTIES FOLDER "src")
target_link_libraries(benchmark_perframe_async PUBLIC ${include_libraries})
target_include_directories(benchmark_perframe_async PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )
                            

add_executable(benchmark_engorgio_async benchmark_engorgio_asynch.cpp
${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
            
set_target_properties(benchmark_engorgio_async PROPERTIES FOLDER "src")
target_link_libraries(benchmark_engorgio_async PUBLIC ${include_libraries})
target_include_directories(benchmark_engorgio_async PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            "${PROJECT_SOURCE_DIR}/library/cxxopts/include/"
                            "${PROJECT_SOURCE_DIR}/library/json/include/"
                            )