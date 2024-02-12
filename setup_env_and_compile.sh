#!/bin/sh

jutil env activate -p mul-tra

# This thing is sourced to load the modules for compilation. Dot is important!
. compile_modules.sh

make clean
make -j8 all
