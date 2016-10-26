# Install script for directory: /home/ritsu/work/parser/lstm-list-parser/cnn/cnn

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/cnn" TYPE FILE FILES
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/aligned-mem-pool.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/c2w.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/cnn.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/conv.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/cuda.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/dict.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/dim.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/exec.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/expr.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/functors.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/gpu-kernels.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/gpu-ops.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/graph.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/gru.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/init.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/lstm.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/model.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/nodes.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/param-nodes.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/random.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/rnn-state-machine.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/rnn.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/saxe-init.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/shadow-params.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/tensor.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/timing.h"
    "/home/ritsu/work/parser/lstm-list-parser/cnn/cnn/training.h"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ritsu/work/parser/lstm-list-parser/build/cnn/cnn/libcnn.a")
endif()

