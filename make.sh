#!/bin/bash

source ./env.sh

echo "CPUS_INCLUDE_PATH = ${CPUS_INCLUDE_PATH}"
echo "C_INCLUDE_PATH    = ${C_INCLUDE_PATH}   "
echo "LD_LIBRARY_PATH   = ${LD_LIBRARY_PATH}  "
echo "PATH   = ${PATH}  "

echo "================================================"
echo " "
make -j 24