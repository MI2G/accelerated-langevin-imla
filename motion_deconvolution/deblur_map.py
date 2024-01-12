import sys
sys.path.append("..")
import os
from ila_utils.sampling_kernels import *
from ila_utils.utils import *
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime
import scipy.io
import torch

from argparse import ArgumentParser

# create a configuration parameter structure
config_parser = ArgumentParser()

# set experiment parameters

# uncomment relevant line

#experiment 1: castle
config_parser.add_argument("--experiment", type=int, default=1)
#experiment 2: person
#config_parser.add_argument("--experiment", type=int, default=2)
#experiment 3: lizard
#config_parser.add_argument("--experiment", type=int, default=3)

config, unknown = config_parser.parse_known_args()

torch.set_default_dtype(torch.float64)
### set seed
torch.manual_seed(0)

#set device
device = 'cuda:0'
torch.set_grad_enabled(False)

### start the script, setup the imaging problem (deblurring)

### use the cvx ridge regularizer
model, L = load_crrnn_and_init(5, 5, device)

# load an image
if config.experiment == 1:
    img, img_torch = load_castle_image(device)
    n_stages = 8
    exp_name = "castle_MAP"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=2)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 2828.4271247461897
    mu = 8.0

elif config.experiment == 2:
    img, img_torch = load_person_image(device)
    n_stages = 12
    exp_name = "person_MAP"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=3)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 2828.4271247461897
    mu = 11.313708498984761

elif config.experiment == 3:
    img, img_torch = load_lizard_image(device)
    n_stages = 8
    exp_name = "lizard_MAP"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=1)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 3363.5856610148585
    mu = 8.0

# setup the operators
A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
    [img.shape[0],img.shape[1]], device, index=3)
print(min_ev, max_ev)

# set the desired noise level and add noise to the burred image
BSNRdb = 30
sigma = torch.linalg.matrix_norm(A(img_torch)-torch.mean(A(img_torch)), ord='fro')/math.sqrt(torch.numel(img_torch)*10**(BSNRdb/10))
sigma2 = sigma**2
img_torch_blurry = A(img_torch) + sigma * torch.randn_like(img_torch)

print(1/sigma2)

### Set up optimizer

# optimization settings
tol = 1e-5
n_iter_max = 200

# stepsize rule
L = model.L
alpha = 1/( AAT_norm/sigma2 + mu * lmbd * L) # with generic H, use 1/(\|HtH\| + mu * lmbd * L)

# initialization
x = torch.clone(img_torch_blurry)
z = torch.clone(img_torch_blurry)
t = 1

restart = True
verbose = True
# optimization loop to compute map
start = time.time()
for i in range(n_iter_max):
    x_old = torch.clone(x)
    grad_z = alpha*(AT(A(x) - img_torch_blurry)/sigma2 + lmbd * model(mu * z))
    x = z - grad_z
    # possible constraint, AGD becomes FISTA
    # e.g. if positivity
    # x = torch.clamp(x, 0, None)
    
    t_old = t 
    t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
    z = x + (t_old - 1)/t * (x - x_old)

    # restart
    if restart and (torch.sum(grad_z*(x - x_old)) > 0):
        t = 1
        z = torch.clone(x)
        if verbose:
            print(f"restart at iteration {i}")
    else:
        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
        if res < tol:
            used_iter = i
            break


end = time.time()
elapsed = end - start
#plot some results

#print pnsr, ssim values
print (f"Number of iterations: {i}, psnr: {peak_signal_noise_ratio(x, img_torch, data_range=1):.3f}")
print (f"ssim: {structural_similarity_index_measure(x, img_torch, data_range=1):.3f}")

x_map = x.detach().cpu().squeeze()

#plot map solution

fig, ax = plt.subplots()
ax.set_title("")
im = ax.imshow(x_map, cmap="gray", vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xticks([])

plt.show()

#save the experiment results

# first, create a dictionary
mdict = {'x_map': x_map.numpy(),
         'n_iter': i,
         'xtrue': img_torch.detach().cpu().squeeze().numpy(),
         'y': img_torch_blurry.detach().cpu().squeeze().numpy(),
         'sigma2' : sigma2.item(),
         'time' : elapsed,
         
        }

# second, setup results directory

date_str = datetime.now().strftime("%Y-%m-%d")
time_str = datetime.now().strftime("%H-%M")
res_path = "results/" + date_str + "_" + exp_name + "/" 


# Check whether the specified path exists or not
if not os.path.exists(res_path):

   # Create a new directory because it does not exist
   os.makedirs(res_path)

#save image
filename_im = res_path + "x_map_" + time_str + ".png"
plt.imsave(filename_im, x_map, cmap="gray")

#save data
mat_name = exp_name + time_str + ".mat"
scipy.io.savemat(res_path + mat_name, 
                  mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')

