# Code for the paper "Accelerated Bayesian imaging by relaxed proximal-point Langevin sampling"
by Teresa Klatzer, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Jesús María Sanz-Serna, Konstantinos C. Zygalakis
https://arxiv.org/abs/2308.09460

Programming languages are Python (Motion deblurring experiments) Matlab (Poisson experiments).

# Preparations

To run the Matlab code, you'll need to install the library to be found in ```libs/L-BFGS-B-C```. Detailed instructions can be found in the original repository by Stephen Becker [here](https://github.com/stephenbeckr/L-BFGS-B-C). The current implementation requires the parallel toolbox, but this is not essential.

To run the Python code, you'll be required to install Python 3.9 and the following packages

```
torch
torchmetrics
tqdm
matplotlib
mpl_toolkits
hdf5storage
PIL
numpy
scipy
```

We recommend the use of CUDA, but it is not required. Watch out for statements like ```device = 'cuda:0'``` and replace them by ```device = 'cpu'``` as required.

Further, we make use of the ```sampling-tools``` package, which can be retrieved and installed from [here](https://github.com/MI2G/sampling-tutorials).

In order to use the [convex ridge regularizer](https://github.com/axgoujon/convex_ridge_regularizers) within the motion deconvolution experiments, I have created a fork [here](https://github.com/axgoujon/convex_ridge_regularizers) and an installable package.
Download the repository and 
```
$ cd cvx_nn_models
$ pip install .
```
The package can then be imported in Python using ```import cvx_nn_models```.


# Motion deconvolution experiments

In the scripts, you can choose the respective experiment (castle, person, lizard) by setting a configuration parameter. The script contains specific hyperparameters for each image and select different blur kernels (see paper for details).

To run the sampling using IMLA, run ```motion-deconvolution/deblur_imla_motion.py```.

To run the sampling using SKROCK, run ```motion-deconvolution/deblur_skrock_motion.py```.

To run the sampling using ULA, run ```motion-deconvolution/deblur_ula_motion.py```.


# Poisson experiments
To run the sampling using the Reflected Implicit Midpoint Algorithm (R-IMLA), run ```poisson/grid_rimla.m```. Results will be saved in an automatically created directory called ```poisson/results```.

To run the sampling using the Reflected SKROCK algorithm, run ```poisson/grid_rskrock.m```. 

To run the sampling using Reflected MYULA, run ```poisson/poisson_deblurring_TV_rmyula.m```.

To run the sampling using Reflected PMALA, run ```poisson/poisson_deblurring_TV_rpmala.m```.

Evaluation scripts and required chains / data to reproduce figures in the paper available upon request (e-mail t.klatzer@sms.ed.ac.uk).
