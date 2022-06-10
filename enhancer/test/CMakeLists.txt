# executables 
list(APPEND include_libraries Compiler Tool Enhancer turbojpeg)

add_executable(testInferEngine test_infer_engine.cpp 
                ${CMAKE_SOURCE_DIR}/library/TensorRT/samples/common/logger.cpp)
set_target_properties(testInferEngine PROPERTIES FOLDER "src")
target_link_libraries(testInferEngine PUBLIC ${include_libraries})
target_include_directories(testInferEngine PUBLIC 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            )

add_executable(testEncodeEngine test_encode_engine.cpp)
set_target_properties(testEncodeEngine PROPERTIES FOLDER "src")
target_link_libraries(testEncodeEngine PUBLIC ${include_libraries})
target_include_directories(testEncodeEngine PUBLIC 
                            "${PROJECT_SOURCE_DIR}/controller/include" 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            )

add_executable(testNeuralEnhancer test_neural_enhancer.cpp)
set_target_properties(testNeuralEnhancer PROPERTIES FOLDER "src")
target_link_libraries(testNeuralEnhancer PUBLIC ${include_libraries})
target_include_directories(testNeuralEnhancer PUBLIC 
                            "${PROJECT_SOURCE_DIR}/controller/include" 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 
                            )




# add_executable(testEncodeEngine_Kakadu test_kakadu_jpeg2000.cpp)
# set_target_properties(testEncodeEngine_Kakadu PROPERTIES FOLDER "src")
# add_compile_definitions(NDEBUG)
# add_compile_definitions(D_LARGEFILE64_SOURCE)
# add_compile_definitions(KDU_X86_INTRINSICS)
# add_compile_definitions(_FILE_OFFSET_BITS=64)
# target_link_libraries(testEncodeEngine_Kakadu PUBLIC ${include_libraries} 
#                     "${PROJECT_SOURCE_DIR}/library/kakadu/libkdu_a82R.so" 
#                     )
# target_include_directories(testEncodeEngine_Kakadu PUBLIC)
		  

add_executable(testlibvpxEngine test_libvpx_engine.cpp
                $ENV{LIBVPX_LIBRARY_PATH}/tools_common.c 
                $ENV{LIBVPX_LIBRARY_PATH}/y4minput.c 
                $ENV{LIBVPX_LIBRARY_PATH}/ivfenc.c 
                $ENV{LIBVPX_LIBRARY_PATH}/video_writer.c 
                $ENV{LIBVPX_LIBRARY_PATH}/vpxstats.c 
                $ENV{LIBVPX_LIBRARY_PATH}/rate_hist.c 
                $ENV{LIBVPX_LIBRARY_PATH}/warnings.c 
                $ENV{LIBVPX_LIBRARY_PATH}/webmenc.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_common.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_any.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/scale_gcc.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_common.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_any.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/row_gcc.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/planar_functions.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libyuv/source/cpu_id.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/common/hdr_util.cc 
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxer.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvwriter.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvparser.cc
                $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvreader.cc 
                )
set_target_properties(testlibvpxEngine PROPERTIES FOLDER "src")
target_link_libraries(testlibvpxEngine PUBLIC ${include_libraries})
target_include_directories(testlibvpxEngine PUBLIC 
                            "${PROJECT_SOURCE_DIR}/controller/include" 
                            "${PROJECT_SOURCE_DIR}/enhancer/include"
                            "${PROJECT_SOURCE_DIR}/tool/include" 
                            "${PROJECT_SOURCE_DIR}/compiler/include" 


                            "${PROJECT_SOURCE_DIR}/library/kakadu" 
                            )
