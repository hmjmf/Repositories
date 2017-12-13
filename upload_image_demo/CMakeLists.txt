cmake_minimum_required(VERSION 3.8)
project(upload_image_demo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message("NO FOUND opencv")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})







set(SOURCE_FILES main.cpp)
add_executable(upload_image_demo ${SOURCE_FILES})
target_link_libraries(upload_image_demo ${OpenCV_LIBS})
