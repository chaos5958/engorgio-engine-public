list(APPEND include_paths "${CMAKE_SOURCE_DIR}/controller/include" 
                          "${CMAKE_SOURCE_DIR}/controller/test" 
                          "${CMAKE_SOURCE_DIR}/tool/include"
                          "${CMAKE_SOURCE_DIR}/enhancer/include"
                          "$ENV{LIBVPX_LIBRARY_PATH}"
                          "$ENV{IPP_LIBRARY_PATH}/include"
                          "$ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm"
                          )
list(APPEND include_libraries Controller Tool Enhancer)
message(${include_libraries})

add_executable(testDecodeEngine test_decode_engine.cpp test_common.cpp
            $ENV{LIBVPX_LIBRARY_PATH}/webmdec.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/common/hdr_util.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxer.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvwriter.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvparser.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvreader.cc 
            )
set_target_properties(testDecodeEngine PROPERTIES FOLDER "test")
target_link_libraries(testDecodeEngine PUBLIC ${include_libraries})
target_include_directories(testDecodeEngine PUBLIC ${include_paths})

add_executable(testAnchorEngine test_anchor_engine.cpp
            test_common.cpp
            $ENV{LIBVPX_LIBRARY_PATH}/webmdec.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/common/hdr_util.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxer.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvmuxer/mkvwriter.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvparser.cc
            $ENV{LIBVPX_LIBRARY_PATH}/third_party/libwebm/mkvparser/mkvreader.cc 
            )
            
set_target_properties(testAnchorEngine PROPERTIES FOLDER "test")
target_link_libraries(testAnchorEngine PUBLIC ${include_libraries})
target_include_directories(testAnchorEngine PUBLIC ${include_paths})