#!/bin/bash

# This file deals with installing BLAS and LAPACK

make_cores=2

lapack_version=3.9.0
lapack_dir=lapack-${lapack_version}

scalapack_version=2.1.0
scalapack_dir=scalapack-${lapack_version}

wget https://github.com/Reference-LAPACK/lapack/archive/v${lapack_version}.tar.gz
tar -xvzf v${lapack_version}.tar.gz

cd ${lapack_dir}

cp make.inc.example make.inc

make -j ${make_cores}
cd BLAS && make -j ${make_cores} && cd ..
cd LAPACKE && make -j ${make_cores} && cd ..
