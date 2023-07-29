#!/bin/bash

set -eux

cd $(dirname $0)/../
mkdir -p build
cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./train
./infer
