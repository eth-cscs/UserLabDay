#!/bin/bash

npx=$1
npy=$2
nptotal=`echo $npx $npy |awk '{print $1*$2+6000}'`

cat << EOF
LAMMPS data file from restart file: timestep = 0, procs = 4

$nptotal atoms

2 atom types

0.0000000000000000e+00 4.0010000000000003e+00 xlo xhi
0.0000000000000000e+00 8.0009999999999994e+00 ylo yhi
-1.0000000000000000e-03 1.0000000000000000e-03 zlo zhi

Masses

1 0.2
2 0.1

Atoms

EOF

#echo ...
cat data.initial.sh.inbc

xmin=0.03
ymin=0.03
xmax=1.0
ymax=2.0
dx=`echo $xmax $npx |awk '{print $1/$2}'`
dy=`echo $ymax $npy |awk '{print $1/$2}'`
ttype=1  # =water
density='1.0e+03'
energy='0.0e+00'
heat='1.0e+00'
zz='0.0e+00'
vv='0 0 0'
id=6000

for jj in `seq 1 $npy` ;do
    for ii in `seq 1 $npx` ;do
        id=`expr $id + 1`
        xx=`echo $ii $dx $xmin |awk '{print ($0-1)*$2 +$3}'`
        yy=`echo $jj $dy $ymin |awk '{print ($0-1)*$2 +$3}'`
        echo "$id $ttype $density $energy $heat $xx $yy $zz $vv"
    done
done

cat << EOF

Velocities

EOF

vv='0.0e+00 0.0e+00 0.0e+00'
for id in `seq 1 $nptotal` ;do
    echo $id $vv
done

echo "# generating data for $np water particles (+ 6000p for boundary conditions)" # > README.JG
echo "# x-axis=[0.03:1.0], y-axis=[0.03:2.0]"                                      #>> README.JG
echo "# $npx particles along x-axis, $npy particles along y-axis"                  #>> README.JG     
echo "# id type rho energy heatcapacity x y z vx vy vz"                            #>> README.JG 
echo "# type1=water type2=bc"                                                      #>> README.JG 
#cat README.JG
