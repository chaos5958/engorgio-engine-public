# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/research/engorgio-engine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/research/engorgio-engine

# Include any dependencies generated for this target.
include eval/CMakeFiles/benchmark_infer.dir/depend.make

# Include the progress variables for this target.
include eval/CMakeFiles/benchmark_infer.dir/progress.make

# Include the compile flags for this target's objects.
include eval/CMakeFiles/benchmark_infer.dir/flags.make

eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o: eval/CMakeFiles/benchmark_infer.dir/flags.make
eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o: eval/benchmark_infer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o"
	cd /workspace/research/engorgio-engine/eval && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o -c /workspace/research/engorgio-engine/eval/benchmark_infer.cpp

eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.i"
	cd /workspace/research/engorgio-engine/eval && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/eval/benchmark_infer.cpp > CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.i

eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.s"
	cd /workspace/research/engorgio-engine/eval && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/eval/benchmark_infer.cpp -o CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.s

eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o: eval/CMakeFiles/benchmark_infer.dir/flags.make
eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o: library/TensorRT/samples/common/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o"
	cd /workspace/research/engorgio-engine/eval && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o -c /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp

eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.i"
	cd /workspace/research/engorgio-engine/eval && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp > CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.i

eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.s"
	cd /workspace/research/engorgio-engine/eval && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp -o CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.s

# Object files for target benchmark_infer
benchmark_infer_OBJECTS = \
"CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o" \
"CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o"

# External object files for target benchmark_infer
benchmark_infer_EXTERNAL_OBJECTS =

eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: eval/CMakeFiles/benchmark_infer.dir/build.make
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: compiler/src/libCompiler.a
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: tool/src/libTool.a
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: enhancer/src/libEnhancer.a
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: library/kakadu/libkdu_a82R.so
eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o: eval/CMakeFiles/benchmark_infer.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/benchmark_infer.dir/cmake_device_link.o"
	cd /workspace/research/engorgio-engine/eval && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_infer.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
eval/CMakeFiles/benchmark_infer.dir/build: eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o

.PHONY : eval/CMakeFiles/benchmark_infer.dir/build

# Object files for target benchmark_infer
benchmark_infer_OBJECTS = \
"CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o" \
"CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o"

# External object files for target benchmark_infer
benchmark_infer_EXTERNAL_OBJECTS =

eval/benchmark_infer: eval/CMakeFiles/benchmark_infer.dir/benchmark_infer.cpp.o
eval/benchmark_infer: eval/CMakeFiles/benchmark_infer.dir/__/library/TensorRT/samples/common/logger.cpp.o
eval/benchmark_infer: eval/CMakeFiles/benchmark_infer.dir/build.make
eval/benchmark_infer: compiler/src/libCompiler.a
eval/benchmark_infer: tool/src/libTool.a
eval/benchmark_infer: enhancer/src/libEnhancer.a
eval/benchmark_infer: library/kakadu/libkdu_a82R.so
eval/benchmark_infer: eval/CMakeFiles/benchmark_infer.dir/cmake_device_link.o
eval/benchmark_infer: eval/CMakeFiles/benchmark_infer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable benchmark_infer"
	cd /workspace/research/engorgio-engine/eval && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmark_infer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
eval/CMakeFiles/benchmark_infer.dir/build: eval/benchmark_infer

.PHONY : eval/CMakeFiles/benchmark_infer.dir/build

eval/CMakeFiles/benchmark_infer.dir/clean:
	cd /workspace/research/engorgio-engine/eval && $(CMAKE_COMMAND) -P CMakeFiles/benchmark_infer.dir/cmake_clean.cmake
.PHONY : eval/CMakeFiles/benchmark_infer.dir/clean

eval/CMakeFiles/benchmark_infer.dir/depend:
	cd /workspace/research/engorgio-engine && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/research/engorgio-engine /workspace/research/engorgio-engine/eval /workspace/research/engorgio-engine /workspace/research/engorgio-engine/eval /workspace/research/engorgio-engine/eval/CMakeFiles/benchmark_infer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : eval/CMakeFiles/benchmark_infer.dir/depend

