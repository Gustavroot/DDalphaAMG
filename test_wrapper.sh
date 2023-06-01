#!/bin/sh
# any error will fail this script
set -e

[ -d testlogs ] || mkdir testlogs
make clean 2>testlogs/cpu_clean.log >testlogs/cpu_clean.log

# Build CPU version of program
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER =/g' MakeSettings.mk
make -j 8 all 2>testlogs/cpu_build.log >testlogs/cpu_build.log
./test/integration_test.sh cpu

make clean 2>testlogs/gpu_clean.log >testlogs/gpu_clean.log
# Build GPU version of program
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER = yes/g' MakeSettings.mk
make -j 8 all 2>testlogs/gpu_build.log >testlogs/gpu_build.log
./test/integration_test.sh gpu

./build/gtest/dd_alpha_amg_gtest |& tee testlogs/gtest.log
