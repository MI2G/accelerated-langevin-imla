# Code for the paper "Accelerated Bayesian imaging by relaxed proximal-point Langevin sampling"
https://arxiv.org/abs/2308.09460

The programming languages are Matlab (Poisson experiments) and Python (Motion deblurring experiments).

# Preparations

To run the Matlab code, you'll need to install the library to be found in ```libs/L-BFGS-B-C```. Detailed instructions can be found in the original repository by Stephen Becker [here](https://github.com/stephenbeckr/L-BFGS-B-C). The current implementation requires the parallel toolbox, but this is not essential.

# Poisson experiments
To run the sampling using the Reflected Implicit Midpoint Algorithm (R-IMLA), run ```poisson/grid_rimla.m```. Results will be saved in an automatically created directory called ```poisson/results```.

To run the sampling using the Reflected SKROCK algorithm, run ```poisson/grid_rskrock.m```. 


Evaluation scripts and required chains / data to reproduce figures in the paper available upon request (e-mail t.klatzer@sms.ed.ac.uk).
