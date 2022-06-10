# executables 
add_executable(test_ipp test_ipp.cpp)

set_target_properties(test_ipp PROPERTIES FOLDER "src")
target_link_libraries(test_ipp PUBLIC ippcore ippcc ippvm ipps ippi)
target_link_directories(test_ipp PUBLIC $ENV{IPP_LIBRARY_PATH}/lib/intel64)
target_include_directories(test_ipp PUBLIC $ENV{IPP_LIBRARY_PATH}/include)

add_executable(test_libturbo_jpeg test_libturbo_jpeg.cpp)
set_target_properties(test_libturbo_jpeg PROPERTIES FOLDER "src")
target_link_libraries(test_libturbo_jpeg PUBLIC pthread turbojpeg)

# gcc ipptest.cpp -o ipptest -I$IPPROOT/include -L$IPPROOT/lib/intel64 -lippcore -lippcc