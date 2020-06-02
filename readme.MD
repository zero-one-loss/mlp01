Stochastic Coordinate Descent for zero-one loss optimization
==

Introduction
-
This repository contains source code to reproduce experiment results
in the paper.

Directory

-   `blackbox_attack` contains the black box attack substitute models' 
definition.
-   `core` contains scd01 and scd01mlp for binary and one-vs-all version
for multi-class.
-   `main` contains training scripts for scd and its variations, svm, mlp-logistic,
training, black-box attack, corruption and perturbation evaluation.
-   `tools` contains args definition, data set loading and other tools
will be used in training scrips.
-   `whitebox_attack` contains script for white-box attack experiments.


Python environment
-
Python v3.6.8

Pytorch v1.3.1

Numpy v1.17.4

Pandas v1.0.3

Argparse v1.1

Sklearn v0.21.2