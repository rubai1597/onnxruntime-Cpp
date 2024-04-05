cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

project(onnxruntime_example)

find_package(CUDA)

if(CUDA_FOUND)
    add_definitions(-DCUDA_FOUND)
endif()

find_package(onnxruntime PATHS /usr/local/onnxruntime/share/cmake)

if(onnxruntime_FOUND)
    message("onnxruntime package FOUND!!!!")
else()
    message("CANNOT find onnxruntime package!!!")

    set(onnxruntime_INSTALL_PATH /usr/local/onnxruntime)

    include_directories(${onnxruntime_INSTALL_PATH}/include)

    file(GLOB onnxruntime_LIBRARIES "${onnxruntime_INSTALL_PATH}/lib/*.so")
endif()

find_package(OpenCV REQUIRED PATHS /usr/local/opencv)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# INCLUDE ADDITIONAL HEADERS
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(
    onnxruntime_example
    main.cc
)

target_link_libraries(
    onnxruntime_example

    dl
    stdc++fs

    ${OpenCV_LIBS}
    ${onnxruntime_LIBRARIES}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/onnxruntime_example)
install(TARGETS onnxruntime_example DESTINATION ./)