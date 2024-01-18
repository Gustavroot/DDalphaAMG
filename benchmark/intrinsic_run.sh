#!/bin/bash

rm ./a.out -rf

gcc -msse -mfma -mavx2 -O3 ./benchmark_simd_intrinsic.c

./a.out
