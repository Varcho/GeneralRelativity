# 🌎 General Relativity Spacetime Visualizer 🌎
(Final Project for OSU Math 5757)

![einstein](https://github.com/Varcho/GeneralRelativity/blob/master/img/einstein.png)

This repo is an implementation/ extension of the methods of techniques descibe [Visualizing Interstellar's Wormhole](https://arxiv.org/pdf/1502.03809.pdf) and [Gravitational Lensing by Spinning...](https://arxiv.org/pdf/1502.03808.pdf)
More specifically, this code implements spacetime path tracing in an ellis-type metric, with the inclusion of an accretion disk. 

Because of the parallelizable nature of path-tracing images, this code was implemented in CUDA, so that it could be efficiently run on the GPU. Additional, multiple integration schemes were implemented (euler, rk4 and rkf45). Although, more computationally costly, rkf45's adaptive nature was necessary for the tracing near the poles, which were prone to numerical inaccuracies.
