# this scripts installs qmp and qio

# TODO : change versions of QMP and QIO !

# FIRST, QMP

mkdir qmp/
cd qmp/

mkdir dir/
cd dir/
QMP_DIR=`pwd`
cd ..

wget http://usqcd.jlab.org/usqcd-software/qmp/qmp-2.1.7.tar.gz
tar -xvzf qmp-2.1.7.tar.gz
cd qmp-2.1.7/

./configure --prefix=$QMP_DIR --with-qmp-comms-type=MPI CC=mpicc

make
make install

# THEN, QIO

cd ../../

mkdir qio/
cd qio/

mkdir dir/
cd dir/
QIO_DIR=`pwd`
cd ..

wget http://usqcd.jlab.org/usqcd-software/qio/qio-2.1.8.tar.gz
tar -xvzf qio-2.1.8.tar.gz
cd qio-2.1.8/

./configure --prefix=$QIO_DIR --with-qmp=$QMP_DIR

make
make install
