# Script/command for creating a sequence of images 'orbiting' around the
# wormhole, thus providing views of the accretion disk from different angles
# Copyright (C) Bill Varcho

set -x 

N=25
for i in `seq 0 ${N}`;
do
 # delegate the arithmatic expresison to bc
  THETA=`bc -l <<< "3.1415926 * ${i} / ${N}"`
  # run the command...
  `../render -s -n ${i} --thetai ${THETA}`;
done 

# open the image directory to view pictures
`open out/`