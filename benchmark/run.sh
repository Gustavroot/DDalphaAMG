#!/bin/bas

rm ./a.out -rf

g++ -mfma -mavx2 -O3 ./benchmark_vectorized_blas.cpp

./a.out
