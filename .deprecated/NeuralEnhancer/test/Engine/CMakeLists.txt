list(APPEND include_paths "${CMAKE_SOURCE_DIR}/NeuralEnhancer/include" 
                            "${CMAKE_SOURCE_DIR}/NeuralEnhancer/test/Common"
                            "${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm")
list(APPEND include_libraries Engine)

add_executable(testDecode testDecode.cpp 
            ${CMAKE_SOURCE_DIR}/library/libvpx/webmdec.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/common/hdr_util.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc
            ${CMAKE_SOURCE_DIR}/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc 
            )
set_target_properties(testDecode PROPERTIES FOLDER "test")
target_link_libraries(testDecode PUBLIC ${include_libraries})
target_include_directories(testDecode PUBLIC ${include_paths})