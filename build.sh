# !/bin/bash

function build(){
  set -e

  ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

  # build rockx
  BUILD_DIR=${ROOT_PWD}/build

  if [[ ! -d "${BUILD_DIR}" ]]; then
    mkdir -p ${BUILD_DIR}
  fi

  cd ${BUILD_DIR}
  cmake .. \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++
  make -j16
  make install
  cd -
}

build