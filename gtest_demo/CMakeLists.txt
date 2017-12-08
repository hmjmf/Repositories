cmake_minimum_required(VERSION 3.8)
project(google_test_demo)


set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(google_test_demo_run ${SOURCE_FILES})


include_directories(add_src)
add_subdirectory(add_src)
add_subdirectory(add_test)

target_link_libraries(google_test_demo_run add_src)