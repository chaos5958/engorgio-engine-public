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
include controller/test/CMakeFiles/testDecodeEngine.dir/depend.make

# Include the progress variables for this target.
include controller/test/CMakeFiles/testDecodeEngine.dir/progress.make

# Include the compile flags for this target's objects.
include controller/test/CMakeFiles/testDecodeEngine.dir/flags.make

controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o: controller/test/test_decode_engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o -c /workspace/research/engorgio-engine/controller/test/test_decode_engine.cpp

controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/test/test_decode_engine.cpp > CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.i

controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/test/test_decode_engine.cpp -o CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.s

controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.o: controller/test/test_common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/test_common.cpp.o -c /workspace/research/engorgio-engine/controller/test/test_common.cpp

controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/test_common.cpp.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/test/test_common.cpp > CMakeFiles/testDecodeEngine.dir/test_common.cpp.i

controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/test_common.cpp.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/test/test_common.cpp -o CMakeFiles/testDecodeEngine.dir/test_common.cpp.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o: library/libvpx/webmdec.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o -c /workspace/research/engorgio-engine/library/libvpx/webmdec.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/webmdec.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/webmdec.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o: library/libvpx/third_party/libwebm/common/hdr_util.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/common/hdr_util.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/common/hdr_util.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/common/hdr_util.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o: library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o: library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o: library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o: library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.s

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o: controller/test/CMakeFiles/testDecodeEngine.dir/flags.make
controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o: library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o"
	cd /workspace/research/engorgio-engine/controller/test && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o -c /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.i"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc > CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.i

controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.s"
	cd /workspace/research/engorgio-engine/controller/test && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc -o CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.s

# Object files for target testDecodeEngine
testDecodeEngine_OBJECTS = \
"CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o" \
"CMakeFiles/testDecodeEngine.dir/test_common.cpp.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o" \
"CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o"

# External object files for target testDecodeEngine
testDecodeEngine_EXTERNAL_OBJECTS =

controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/test_decode_engine.cpp.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/test_common.cpp.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/webmdec.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/common/hdr_util.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxer.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvmuxerutil.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvmuxer/mkvwriter.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvparser.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/__/__/library/libvpx/third_party/libwebm/mkvparser/mkvreader.cc.o
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/build.make
controller/test/testDecodeEngine: controller/src/libController.a
controller/test/testDecodeEngine: tool/src/libTool.a
controller/test/testDecodeEngine: controller/test/CMakeFiles/testDecodeEngine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable testDecodeEngine"
	cd /workspace/research/engorgio-engine/controller/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testDecodeEngine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
controller/test/CMakeFiles/testDecodeEngine.dir/build: controller/test/testDecodeEngine

.PHONY : controller/test/CMakeFiles/testDecodeEngine.dir/build

controller/test/CMakeFiles/testDecodeEngine.dir/clean:
	cd /workspace/research/engorgio-engine/controller/test && $(CMAKE_COMMAND) -P CMakeFiles/testDecodeEngine.dir/cmake_clean.cmake
.PHONY : controller/test/CMakeFiles/testDecodeEngine.dir/clean

controller/test/CMakeFiles/testDecodeEngine.dir/depend:
	cd /workspace/research/engorgio-engine && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/research/engorgio-engine /workspace/research/engorgio-engine/controller/test /workspace/research/engorgio-engine /workspace/research/engorgio-engine/controller/test /workspace/research/engorgio-engine/controller/test/CMakeFiles/testDecodeEngine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : controller/test/CMakeFiles/testDecodeEngine.dir/depend

