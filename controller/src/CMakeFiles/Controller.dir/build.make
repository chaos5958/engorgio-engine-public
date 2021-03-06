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
include controller/src/CMakeFiles/Controller.dir/depend.make

# Include the progress variables for this target.
include controller/src/CMakeFiles/Controller.dir/progress.make

# Include the compile flags for this target's objects.
include controller/src/CMakeFiles/Controller.dir/flags.make

controller/src/CMakeFiles/Controller.dir/controller.cpp.o: controller/src/CMakeFiles/Controller.dir/flags.make
controller/src/CMakeFiles/Controller.dir/controller.cpp.o: controller/src/controller.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object controller/src/CMakeFiles/Controller.dir/controller.cpp.o"
	cd /workspace/research/engorgio-engine/controller/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Controller.dir/controller.cpp.o -c /workspace/research/engorgio-engine/controller/src/controller.cpp

controller/src/CMakeFiles/Controller.dir/controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Controller.dir/controller.cpp.i"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/src/controller.cpp > CMakeFiles/Controller.dir/controller.cpp.i

controller/src/CMakeFiles/Controller.dir/controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Controller.dir/controller.cpp.s"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/src/controller.cpp -o CMakeFiles/Controller.dir/controller.cpp.s

controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.o: controller/src/CMakeFiles/Controller.dir/flags.make
controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.o: controller/src/decode_engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.o"
	cd /workspace/research/engorgio-engine/controller/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Controller.dir/decode_engine.cpp.o -c /workspace/research/engorgio-engine/controller/src/decode_engine.cpp

controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Controller.dir/decode_engine.cpp.i"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/src/decode_engine.cpp > CMakeFiles/Controller.dir/decode_engine.cpp.i

controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Controller.dir/decode_engine.cpp.s"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/src/decode_engine.cpp -o CMakeFiles/Controller.dir/decode_engine.cpp.s

controller/src/CMakeFiles/Controller.dir/control_common.cpp.o: controller/src/CMakeFiles/Controller.dir/flags.make
controller/src/CMakeFiles/Controller.dir/control_common.cpp.o: controller/src/control_common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object controller/src/CMakeFiles/Controller.dir/control_common.cpp.o"
	cd /workspace/research/engorgio-engine/controller/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Controller.dir/control_common.cpp.o -c /workspace/research/engorgio-engine/controller/src/control_common.cpp

controller/src/CMakeFiles/Controller.dir/control_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Controller.dir/control_common.cpp.i"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/src/control_common.cpp > CMakeFiles/Controller.dir/control_common.cpp.i

controller/src/CMakeFiles/Controller.dir/control_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Controller.dir/control_common.cpp.s"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/src/control_common.cpp -o CMakeFiles/Controller.dir/control_common.cpp.s

controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.o: controller/src/CMakeFiles/Controller.dir/flags.make
controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.o: controller/src/anchor_engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.o"
	cd /workspace/research/engorgio-engine/controller/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Controller.dir/anchor_engine.cpp.o -c /workspace/research/engorgio-engine/controller/src/anchor_engine.cpp

controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Controller.dir/anchor_engine.cpp.i"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/controller/src/anchor_engine.cpp > CMakeFiles/Controller.dir/anchor_engine.cpp.i

controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Controller.dir/anchor_engine.cpp.s"
	cd /workspace/research/engorgio-engine/controller/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/controller/src/anchor_engine.cpp -o CMakeFiles/Controller.dir/anchor_engine.cpp.s

# Object files for target Controller
Controller_OBJECTS = \
"CMakeFiles/Controller.dir/controller.cpp.o" \
"CMakeFiles/Controller.dir/decode_engine.cpp.o" \
"CMakeFiles/Controller.dir/control_common.cpp.o" \
"CMakeFiles/Controller.dir/anchor_engine.cpp.o"

# External object files for target Controller
Controller_EXTERNAL_OBJECTS =

controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/controller.cpp.o
controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/decode_engine.cpp.o
controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/control_common.cpp.o
controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/anchor_engine.cpp.o
controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/build.make
controller/src/libController.a: controller/src/CMakeFiles/Controller.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libController.a"
	cd /workspace/research/engorgio-engine/controller/src && $(CMAKE_COMMAND) -P CMakeFiles/Controller.dir/cmake_clean_target.cmake
	cd /workspace/research/engorgio-engine/controller/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Controller.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
controller/src/CMakeFiles/Controller.dir/build: controller/src/libController.a

.PHONY : controller/src/CMakeFiles/Controller.dir/build

controller/src/CMakeFiles/Controller.dir/clean:
	cd /workspace/research/engorgio-engine/controller/src && $(CMAKE_COMMAND) -P CMakeFiles/Controller.dir/cmake_clean.cmake
.PHONY : controller/src/CMakeFiles/Controller.dir/clean

controller/src/CMakeFiles/Controller.dir/depend:
	cd /workspace/research/engorgio-engine && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/research/engorgio-engine /workspace/research/engorgio-engine/controller/src /workspace/research/engorgio-engine /workspace/research/engorgio-engine/controller/src /workspace/research/engorgio-engine/controller/src/CMakeFiles/Controller.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : controller/src/CMakeFiles/Controller.dir/depend

