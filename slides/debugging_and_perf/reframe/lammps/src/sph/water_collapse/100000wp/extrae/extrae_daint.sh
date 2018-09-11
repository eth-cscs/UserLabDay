#!/bin/bash

source $EBROOTEXTRAE/etc/extrae.sh
export EXTRAE_CONFIG_FILE=`pwd`/extrae/extrae354.xml

#c+mpi:
export LD_PRELOAD=$EBROOTEXTRAE/lib/libmpitrace.so

#f90+mpi:
#export LD_PRELOAD=$EXTRAE_HOME/lib/libmpitracef.so

#f90+mpi/openmp: 
#export LD_PRELOAD=$EBROOTEXTRAE/lib/libompitracef.so
#export LD_PRELOAD=$EXTRAE_HOME/lib/libompitracef.so

## Run the desired program
lmp_mpi -in water_collapse.lmp
## $* $@
