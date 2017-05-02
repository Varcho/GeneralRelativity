# Script/command for creating a sequence of images 'orbiting' around the
# wormhole, thus providing views of the accretion disk from different angles
# Copyright (C) Bill Varcho

import subprocess
N = 25
for i in range(N):
	command = "../render -s -n %i --thetai %f" %(i,3.14159*float(i)/float(N))
	subprocess.call([command, ""], shell=True)
