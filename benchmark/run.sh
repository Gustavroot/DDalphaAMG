#!/bin/bash

rm ./a.out -rf

g++ $1 $2 $3 -msse -mfma -mavx2 -O3 ./benchmark_vectorized_blas.cpp

./a.out
