#!/bin/bash

clean() {
    echo "Cleaning up..."
    ./cleanfile.sh
    rm -rf ./m1_cpu ./m1_gpu ./m2_unroll ./m2 ./m3 ./viz *.out *.err outfile
}

build() {
    echo "Building the project..."
    ./cleanfile.sh
    cmake -DCMAKE_BUILD_TYPE=Release ./project/ && make -j"$(nproc)"
    ./cleanfile.sh
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
