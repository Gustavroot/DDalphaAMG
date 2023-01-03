#!/bin/sh
# any error will fail this script
set -e

make clean 2>build/cpu_clean.log >build/cpu_clean.log
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER =/g' MakeSettings.mk
make -j 8 all 2>build/cpu_build.log >build/cpu_build.log
./test/integration_test.sh cpu

make clean 2>build/gpu_clean.log >build/gpu_clean.log
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER = -DCUDA_OPT/g' MakeSettings.mk
make -j 8 all 2>build/gpu_build.log >build/gpu_build.log
./test/integration_test.sh gpu