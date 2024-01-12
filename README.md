# Code for the paper "Accelerated Bayesian imaging by relaxed proximal-point Langevin sampling"
by Teresa Klatzer, Paul Dobson, Yoann Altmann, Marcelo Pereyra, Jesús María Sanz-Serna, Konstantinos C. Zygalakis
https://arxiv.org/abs/2308.09460

### Abstract
This paper presents a new accelerated proximal Markov chain Monte Carlo methodology to perform Bayesian inference in imaging inverse problems with an underlying convex geometry. The proposed strategy takes the form of a stochastic relaxed proximal-point iteration that admits two complementary interpretations. For models that are smooth or regularised by Moreau-Yosida smoothing,
the algorithm is equivalent to an implicit midpoint discretisation of an overdamped Langevin diffusion targeting the posterior distribution of interest. This discretisation is asymptotically unbiased for
Gaussian targets and shown to converge in an accelerated manner for any target that is κ-strongly
log-concave (i.e., requiring in the order of √
κ iterations to converge, similarly to accelerated optimisation schemes), comparing favorably to [M. Pereyra, L. Vargas Mieles, K.C. Zygalakis, SIAM
J. Imaging Sciences, 13,2 (2020), pp. 905-935] which is only provably accelerated for Gaussian
targets and has bias. For models that are not smooth, the algorithm is equivalent to a Leimkuhler–Matthews discretisation of a Langevin diffusion targeting a Moreau-Yosida approximation of the
posterior distribution of interest, and hence achieves a significantly lower bias than conventional
unadjusted Langevin strategies based on the Euler-Maruyama discretisation. For targets that are
κ-strongly log-concave, the provided non-asymptotic convergence analysis also identifies the optimal time step which maximizes the convergence speed. The proposed methodology is demonstrated
through a range of experiments related to image deconvolution with Gaussian and Poisson noise,
with assumption-driven and data-driven convex priors. 

### Languages
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

To run the sampling using IMLA, run ```motion_deconvolution/deblur_imla_motion.py```.

To run the sampling using SKROCK, run ```motion_deconvolution/deblur_skrock_motion.py```.

To run the sampling using ULA, run ```motion_deconvolution/deblur_ula_motion.py```.

To run the sampling using PnP-ULA, run ```motion_deconvolution/PnP_ULA_motion.py```.

To compute the MAP solution, run ```motion_deconvolution/deblur_map.py```.



# Poisson experiments
To run the sampling using the Reflected Implicit Midpoint Algorithm (R-IMLA), run ```poisson/grid_rimla.m```. Results will be saved in an automatically created directory called ```poisson/results```.

To run the sampling using the Reflected SKROCK algorithm, run ```poisson/grid_rskrock.m```. 

To run the sampling using Reflected MYULA, run ```poisson/poisson_deblurring_TV_rmyula.m```.

To run the sampling using Reflected PMALA, run ```poisson/poisson_deblurring_TV_rpmala.m```.

Evaluation scripts and required chains / data to reproduce figures in the paper available upon request (e-mail t.klatzer@sms.ed.ac.uk).

# 1D experiments

Code to sample distributions for Figure 3 can be found in ```one_d_examples/one_d_prox.m```.

# Citation

If you find our code helpful in your resarch or work, please cite our [paper](https://arxiv.org/abs/2308.09460).
