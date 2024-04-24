#!/bin/bash

# This file deals with installing BLACS and ScaLAPACK

echo "READ THE CONTENT OF THIS FILE AND FOLLOW THE INSTRUCTIONS"

: <<'END'

PRE-WORK : check if <f77> is an available command, and if not, create a symbolic link
           to <gfortran> with that name


1. download scalapack installer

wget http://www.netlib.org/scalapack/scalapack_installer.tgz
tar -xvzf scalapack_installer.tgz
mv scalapack_installer scalapack
cd scalapack/


2. do the following changes :

----
script/lapack.py :

LINE 258 :: os.chdir(os.path.join(os.getcwd(),'lapack-3.9.0'))

LINES 367-369 ::
        # move lib to the lib directory
        shutil.copy('SRC/libreflapack.a',os.path.join(self.prefix,'lib/libreflapack.a'))
        shutil.copy('TESTING/MATGEN/libtmg.a',os.path.join(self.prefix,'lib/libtmg.a'))

COMMENT FROM LINE 371 ONWARDS

ADD THIS LINE AT THE VERY END : os.chdir(savecwd)

----
script/framework.py :

LINE 31 :: scalapackurl = 'http://www.netlib.org/scalapack/scalapack-2.1.0.tgz'

---
script/scalapack.py :

27  :: testing      = 0                         # whether on not to compile and run LAPACK and ScaLAPACK test programs
114 :: comm = 'gunzip -f scalapack-2.1.0.tgz'
122 :: comm = 'tar xf scalapack-2.1.0.tar'
128 :: os.remove('scalapack-2.1.0.tar')


3. finally, run :

./setup.py --prefix=/home/ramirez/Documents/DDalphaAMG_ci/dependencies/scalapack --mpicc=/usr/lib64/mpi/gcc/openmpi/bin/mpicc --mpiincdir=/usr/lib64/mpi/gcc/openmpi/include/ --mpif90="/usr/lib64/mpi/gcc/openmpi/bin/mpif90" --downall

END
