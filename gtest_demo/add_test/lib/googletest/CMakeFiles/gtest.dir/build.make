# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.8.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.8.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/qixiangzhang/Desktop/testcode/google_test_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/qixiangzhang/Desktop/testcode/google_test_demo

# Include any dependencies generated for this target.
include add_test/lib/googletest/CMakeFiles/gtest.dir/depend.make

# Include the progress variables for this target.
include add_test/lib/googletest/CMakeFiles/gtest.dir/progress.make

# Include the compile flags for this target's objects.
include add_test/lib/googletest/CMakeFiles/gtest.dir/flags.make

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o: add_test/lib/googletest/CMakeFiles/gtest.dir/flags.make
add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o: add_test/lib/googletest/src/gtest-all.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/qixiangzhang/Desktop/testcode/google_test_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o"
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gtest.dir/src/gtest-all.cc.o -c /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest/src/gtest-all.cc

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gtest.dir/src/gtest-all.cc.i"
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest/src/gtest-all.cc > CMakeFiles/gtest.dir/src/gtest-all.cc.i

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gtest.dir/src/gtest-all.cc.s"
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest/src/gtest-all.cc -o CMakeFiles/gtest.dir/src/gtest-all.cc.s

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires:

.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides: add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires
	$(MAKE) -f add_test/lib/googletest/CMakeFiles/gtest.dir/build.make add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides.build
.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides

add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.provides.build: add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o


# Object files for target gtest
gtest_OBJECTS = \
"CMakeFiles/gtest.dir/src/gtest-all.cc.o"

# External object files for target gtest
gtest_EXTERNAL_OBJECTS =

add_test/lib/googletest/libgtest.a: add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
add_test/lib/googletest/libgtest.a: add_test/lib/googletest/CMakeFiles/gtest.dir/build.make
add_test/lib/googletest/libgtest.a: add_test/lib/googletest/CMakeFiles/gtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/qixiangzhang/Desktop/testcode/google_test_demo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgtest.a"
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest.dir/cmake_clean_target.cmake
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
add_test/lib/googletest/CMakeFiles/gtest.dir/build: add_test/lib/googletest/libgtest.a

.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/build

add_test/lib/googletest/CMakeFiles/gtest.dir/requires: add_test/lib/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o.requires

.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/requires

add_test/lib/googletest/CMakeFiles/gtest.dir/clean:
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest && $(CMAKE_COMMAND) -P CMakeFiles/gtest.dir/cmake_clean.cmake
.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/clean

add_test/lib/googletest/CMakeFiles/gtest.dir/depend:
	cd /Users/qixiangzhang/Desktop/testcode/google_test_demo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/qixiangzhang/Desktop/testcode/google_test_demo /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest /Users/qixiangzhang/Desktop/testcode/google_test_demo /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest /Users/qixiangzhang/Desktop/testcode/google_test_demo/add_test/lib/googletest/CMakeFiles/gtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : add_test/lib/googletest/CMakeFiles/gtest.dir/depend

