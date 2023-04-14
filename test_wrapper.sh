#!/bin/sh
# any error will fail this script
set -e

[ -d testlogs ] || mkdir testlogs
make clean 2>testlogs/cpu_clean.log >testlogs/cpu_clean.log
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER =/g' MakeSettings.mk
make -j 8 all 2>testlogs/cpu_build.log >testlogs/cpu_build.log
./test/integration_test.sh cpu

[ -d testlogs ] || mkdir testlogs
make clean 2>testlogs/gpu_clean.log >testlogs/gpu_clean.log
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER = -DCUDA_OPT/g' MakeSettings.mk
make -j 8 all 2>testlogs/gpu_build.log >testlogs/gpu_build.log
./test/integration_test.sh gpu
