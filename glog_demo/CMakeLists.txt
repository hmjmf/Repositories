cmake_minimum_required(VERSION 3.8)
project(glog_test)

set(CMAKE_CXX_STANDARD 11)

find_package(Glog REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(glog_test ${SOURCE_FILES})
target_link_libraries(glog_test glog::glog)