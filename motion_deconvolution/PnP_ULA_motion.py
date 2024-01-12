
# %%
import sys
sys.path.append("..")
import os
import math
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import time as time
from tqdm.auto import tqdm
import scipy.io
from datetime import datetime
from ila_utils.sampling_kernels import *
from ila_utils.utils import *
from ila_utils.Spectral_Normalize_chen import spectral_norm

from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM

import sampling_tools as st

import matplotlib.pyplot as plt

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

### Helpers
### necessary to use the PnP network within ULA
### some code taken from the author's repository published here https://github.com/uclaopt/Provable_Plug_and_Play
### E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin. "Plug-and-Play Methods Provably Converge with Properly Trained Denoisers." ICML, 2019.

import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x).detach()
        return out

# ---- load the model based on the type and sigma (noise level) ---- 
def load_model(model_type, sigma,device):

    path = "../pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"

    net = DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net, device_ids=[device]).cuda(device)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model

# Check if there's a GPU available and run on GPU, otherwise run on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load an image
if config.experiment == 1:
    img, img_torch = load_castle_image(device)
    n_stages = 8
    exp_name = "castle_PnP_ULA"
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
    exp_name = "person_PnP_ULA"
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
    exp_name = "lizard_PnP_ULA"
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


# set the desired noise level and add noise to the burred image
BSNRdb = 30
sigma = torch.linalg.matrix_norm(A(img_torch)-torch.mean(A(img_torch)), ord='fro')/math.sqrt(torch.numel(img_torch)*10**(BSNRdb/10))
sigma2 = sigma**2
img_torch_blurry = A(img_torch) + sigma * torch.randn_like(img_torch)

plt.imshow(img_torch_blurry.detach().cpu().squeeze(), cmap='gray')
plt.title("noisy and blurry observation")


# Define some lambda functions
f = lambda x,A : (torch.linalg.matrix_norm(x-A(x), ord='fro')**2.0)/(2.0*sigma**2)
gradf = lambda x,A,AT : AT(A(x)-img_torch_blurry)/sigma**2

# Set Lipschitz constants
L_y = AAT_norm/(sigma**2)
L_net = 1.0

# load network model and set denoiser 
model = load_model("RealSN_DnCNN", 5, device)
denoise = lambda x: (x - model(x)).detach() 

# algorithm parameters
alpha = 1
eps =  (5/255)**2
max_lambd = 1.0/((2.0*alpha*L_net)/eps+4.0*L_y)
lambd_frac = 0.99
lambd = max_lambd*lambd_frac

C_upper_lim = torch.tensor(1).to(device)
C_lower_lim = torch.tensor(0).to(device)

# ### PnP-ULA and PPnP-ULA kernel updates
projbox = lambda x: torch.clamp(x, min = C_lower_lim, max = C_upper_lim)

def Markov_kernel(x, delta, projected):
    if projected:
        return projbox(x - delta * gradf(x,A,AT) + alpha*delta/eps*(denoise(x)-x) + math.sqrt(2*delta) * torch.randn_like(x))
    else:
        return x - delta * gradf(x,A,AT) + alpha*delta/eps*(denoise(x)-x) + delta/lambd*(projbox(x)-x) + math.sqrt(2*delta) * torch.randn_like(x)

# ### Setting the stepsize

projected = True

if projected:
    delta_max = (1.0)/(L_net/eps+L_y)
else:
    delta_max = (1.0/3.0)/((alpha*L_net)/eps+L_y+1/lambd)
delta_frac = 0.99
delta = delta_max*delta_frac

# Set iterations and prepare sampling
maxit = 480000
burnin = np.int64(maxit*0.05)
n_samples = np.int64(2000)
x = img_torch_blurry.clone()
MC_x = []
thinned_trace_counter = 0
thinning_step = np.int64(maxit/n_samples)

# ### Quality metrics
# Keep track of the PSNR, SSIM w.r.t. to the ground truth image and the log-posterior on the fly.
psnr_values = []
ssim_values = []


start_time = time.time()
for i in tqdm(range(maxit)):

    # Update x
    x = Markov_kernel(x, delta, projected=projected)

    if i == burnin:
        # Initialise recording of sample summary statistics after burnin period
        post_meanvar = st.welford(x)
        absfouriercoeff = st.welford(torch.fft.fft2(x).abs())
        count=0
    elif i > burnin:
        # update the sample summary statistics
        post_meanvar.update(x)
        absfouriercoeff.update(torch.fft.fft2(x).abs())

        # collect quality measurements
        current_mean = post_meanvar.get_mean()
        psnr_values.append(PSNR(img_torch, current_mean).item())
        ssim_values.append(SSIM(img_torch, current_mean).item())

        # collect thinned trace
        if count == thinning_step-1:
            MC_x.append(x.detach().cpu().numpy())
            count = 0
        else:
            count += 1

end_time = time.time()
elapsed = end_time - start_time       


# Print some stats
print(f"Initial PSNR: {PSNR(img_torch,img_torch_blurry):.2f} dB")
print(f"Initial SSIM: {SSIM(img_torch,img_torch_blurry):.4f}")

print(f"Result PSNR: {PSNR(post_meanvar.get_mean(),img_torch):.2f} dB")
print(f"Result SSIM: {SSIM(post_meanvar.get_mean(),img_torch):.4f}")


# Plot some images
plt.figure()
plt.imshow(x.detach().cpu().squeeze(), cmap='gray')
plt.title("reconstructed x")

plt.figure()
plt.plot(np.arange(len(psnr_values)), psnr_values)
plt.title("PSNR")


plt.show()

#save the experiment results

# first, create a dictionary
mdict = {'current_iter': maxit,
         'running_mean': post_meanvar.get_mean().detach().cpu().squeeze().numpy(),
         'running_var_n': post_meanvar.get_var().detach().cpu().squeeze().numpy(),
         'running_mean_fft': absfouriercoeff.get_mean().detach().cpu().squeeze().numpy(),
         'running_var_n_fft': absfouriercoeff.get_var().detach().cpu().squeeze().numpy(),
         'xtrue': img_torch.detach().cpu().squeeze().numpy(),
         'y': img_torch_blurry.detach().cpu().squeeze().numpy(),
         'trace_x': np.array(MC_x),
         'thinning': thinning_step,
         'trace_psnr': np.array(psnr_values),
         'trace_ssim': np.array(ssim_values),
         'sigma2' : sigma2.item(),
         'L': (L_net/eps+L_y).item(),
         'time' : elapsed,
        }

# second, setup results directory

date_str = datetime.now().strftime("%Y-%m-%d")
res_path = "results/" + date_str + "_" + exp_name + "/" 


# Check whether the specified path exists or not
if not os.path.exists(res_path):

   # Create a new directory because it does not exist
   os.makedirs(res_path)

#save image
filename_im = res_path + "x_recon.png"
plt.imsave(filename_im, x.detach().cpu().squeeze(), cmap="gray")

mat_name = exp_name + ".mat"
scipy.io.savemat(res_path + mat_name, 
                  mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


