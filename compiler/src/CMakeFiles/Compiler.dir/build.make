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
include compiler/src/CMakeFiles/Compiler.dir/depend.make

# Include the progress variables for this target.
include compiler/src/CMakeFiles/Compiler.dir/progress.make

# Include the compile flags for this target's objects.
include compiler/src/CMakeFiles/Compiler.dir/flags.make

compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.o: compiler/src/CMakeFiles/Compiler.dir/flags.make
compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.o: compiler/src/compile_engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.o"
	cd /workspace/research/engorgio-engine/compiler/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Compiler.dir/compile_engine.cpp.o -c /workspace/research/engorgio-engine/compiler/src/compile_engine.cpp

compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Compiler.dir/compile_engine.cpp.i"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/compiler/src/compile_engine.cpp > CMakeFiles/Compiler.dir/compile_engine.cpp.i

compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Compiler.dir/compile_engine.cpp.s"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/compiler/src/compile_engine.cpp -o CMakeFiles/Compiler.dir/compile_engine.cpp.s

compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.o: compiler/src/CMakeFiles/Compiler.dir/flags.make
compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.o: compiler/src/compile_common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.o"
	cd /workspace/research/engorgio-engine/compiler/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Compiler.dir/compile_common.cpp.o -c /workspace/research/engorgio-engine/compiler/src/compile_common.cpp

compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Compiler.dir/compile_common.cpp.i"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/compiler/src/compile_common.cpp > CMakeFiles/Compiler.dir/compile_common.cpp.i

compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Compiler.dir/compile_common.cpp.s"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/compiler/src/compile_common.cpp -o CMakeFiles/Compiler.dir/compile_common.cpp.s

compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o: compiler/src/CMakeFiles/Compiler.dir/flags.make
compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o: library/TensorRT/samples/common/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o"
	cd /workspace/research/engorgio-engine/compiler/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o -c /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp

compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.i"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp > CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.i

compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.s"
	cd /workspace/research/engorgio-engine/compiler/src && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/research/engorgio-engine/library/TensorRT/samples/common/logger.cpp -o CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.s

# Object files for target Compiler
Compiler_OBJECTS = \
"CMakeFiles/Compiler.dir/compile_engine.cpp.o" \
"CMakeFiles/Compiler.dir/compile_common.cpp.o" \
"CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o"

# External object files for target Compiler
Compiler_EXTERNAL_OBJECTS =

compiler/src/libCompiler.a: compiler/src/CMakeFiles/Compiler.dir/compile_engine.cpp.o
compiler/src/libCompiler.a: compiler/src/CMakeFiles/Compiler.dir/compile_common.cpp.o
compiler/src/libCompiler.a: compiler/src/CMakeFiles/Compiler.dir/__/__/library/TensorRT/samples/common/logger.cpp.o
compiler/src/libCompiler.a: compiler/src/CMakeFiles/Compiler.dir/build.make
compiler/src/libCompiler.a: compiler/src/CMakeFiles/Compiler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/research/engorgio-engine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libCompiler.a"
	cd /workspace/research/engorgio-engine/compiler/src && $(CMAKE_COMMAND) -P CMakeFiles/Compiler.dir/cmake_clean_target.cmake
	cd /workspace/research/engorgio-engine/compiler/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Compiler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
compiler/src/CMakeFiles/Compiler.dir/build: compiler/src/libCompiler.a

.PHONY : compiler/src/CMakeFiles/Compiler.dir/build

compiler/src/CMakeFiles/Compiler.dir/clean:
	cd /workspace/research/engorgio-engine/compiler/src && $(CMAKE_COMMAND) -P CMakeFiles/Compiler.dir/cmake_clean.cmake
.PHONY : compiler/src/CMakeFiles/Compiler.dir/clean

compiler/src/CMakeFiles/Compiler.dir/depend:
	cd /workspace/research/engorgio-engine && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/research/engorgio-engine /workspace/research/engorgio-engine/compiler/src /workspace/research/engorgio-engine /workspace/research/engorgio-engine/compiler/src /workspace/research/engorgio-engine/compiler/src/CMakeFiles/Compiler.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : compiler/src/CMakeFiles/Compiler.dir/depend

