cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)

find_package(Glog REQUIRED)

project(server)
set(SERVER_SOURCE_FILES server_main.cpp help.cpp)
add_executable(server ${SERVER_SOURCE_FILES})
target_link_libraries(server glog::glog)

project(client)
set(CLIENT_SOURCE_FILES client_main.cpp  help.cpp)
add_executable(client ${CLIENT_SOURCE_FILES})
target_link_libraries(client glog::glog)