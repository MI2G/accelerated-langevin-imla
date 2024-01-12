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
from tqdm.auto import tqdm
import sampling_tools as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

### set dtype
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
    exp_name = "castle_IMLA_"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=2)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 2828.4271247461897
    mu = 8.0

elif config.experiment == 2:
    img, img_torch = load_person_image(device)
    exp_name = "person_IMLA_"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=3)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 2828.4271247461897
    mu = 11.313708498984761

elif config.experiment == 3:
    img, img_torch = load_lizard_image(device)
    exp_name = "lizard_IMLA_"
    # setup the operators
    A, AT, AAT_norm, min_ev, max_ev = motion_blur_operators(
        [img.shape[0],img.shape[1]], device, index=1)
    # setup hyperparameters for the network regularizer
    # found by grid search
    lmbd = 3363.5856610148585
    mu = 8.0

print(min_ev, max_ev)

# set the desired noise level and add noise to the burred image
BSNRdb = 30
sigma = torch.linalg.matrix_norm(A(img_torch)-torch.mean(A(img_torch)), ord='fro')/math.sqrt(torch.numel(img_torch)*10**(BSNRdb/10))
sigma2 = sigma**2
img_torch_blurry = A(img_torch) + sigma * torch.randn_like(img_torch)

print(1/sigma2)

### f and gradf for network
f = lambda x: lmbd/mu * model.cost(mu * x) + 0.5*torch.norm((A(x) - img_torch_blurry))**2/sigma2
grad_f = lambda x: AT(A(x) - img_torch_blurry)/sigma2 + lmbd * model(mu * x)

### Now, try IMLA
# Set up sampler
## Lipschitz constant of the problem
Lip_total = mu * lmbd * L + AAT_norm/sigma2

print("m ", min_ev)
print("M ", max_ev)

m = min_ev/sigma2

#set optimal step size for problem
h = 2/torch.sqrt(Lip_total*m)

# how many sampling iterations to run
n_iter = 40000
theta = 0.5

# counter for thinned trace
cnt = 0

# max 2000 samples in trace
save_iter = np.floor(n_iter/np.minimum(2000,n_iter))
# length of thinned data
trace_len = int(n_iter/save_iter)

# set up some vectors to collect statistics
# used every iter
logpi_vals = torch.zeros((n_iter, 1))
# used only every save_iter
psnr_vals =  torch.zeros((trace_len,1))
ssim_vals = torch.zeros((trace_len,1))
samples = np.zeros((trace_len,) + img_torch.shape)

# initialize variables
ila_sample = img_torch_blurry
post_var = st.welford(ila_sample)
post_fft_var = st.welford(torch.real(torch.fft.fft2(ila_sample)))
post_abs_fft_var = st.welford(torch.fft.fft2(ila_sample).abs())

mean = post_var.get_mean()
var = post_var.get_mean()

exp_name += str(n_iter) + "n_iter"

start = time.time()
# sampling loop
for i in tqdm(range(n_iter)):

    Z_k = torch.randn_like(img_torch_blurry)
    ila_sample = ila_one_step(ila_sample, Z_k, theta, h, f,
                                tolerance=1e-5, hist_size=5, device=device)

    #collect stats every iter
    logpi_vals[i] = f(ila_sample).item()
    post_var.update(ila_sample)
    post_fft_var.update(torch.real(torch.fft.fft2(ila_sample)))
    post_abs_fft_var.update(torch.fft.fft2(ila_sample).abs())

    #collect stats every "save_iter"
    if np.mod(i, save_iter) == 0:
        samples[cnt] = ila_sample.detach().clone().cpu()
        mean = post_var.get_mean()
        var = post_var.get_var()
        mean_fft = post_fft_var.get_mean()
        var_fft = post_fft_var.get_var()
        mean_abs_fft = post_abs_fft_var.get_mean()
        var_abs_fft = post_abs_fft_var.get_var()
        psnr_vals[cnt] = peak_signal_noise_ratio(mean, img_torch, data_range=1.0).item()
        ssim_vals[cnt] = structural_similarity_index_measure(mean, img_torch, data_range=1.0).item()

        cnt += 1


end = time.time()
elapsed = end - start

#plot some results

plt.figure()
plt.loglog(torch.tensor(range(n_iter+1)[1:]).cpu(), logpi_vals.cpu())
plt.title("logpi plot")

plt.figure()
plt.plot(psnr_vals)
plt.title("PSNR")
plt.legend(["imla"])

plt.figure()
plt.plot(ssim_vals)
plt.title("SSIM")
plt.legend(["imla"])

fig, ax = plt.subplots()
ax.set_title(f"Deblurred Image IMLA MMSE (Regularization Cost {model.cost(mu*mean)[0].item():.1f}, PSNR: {peak_signal_noise_ratio(mean, img_torch, data_range=1.0).item():.2f})")
im = ax.imshow(mean.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
ax.set_yticks([])
ax.set_xticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

fig, ax = plt.subplots()
ax.set_title(f"IMLA variance")
im = ax.imshow(var.detach().cpu().squeeze(), cmap="gray")
ax.set_yticks([])
ax.set_xticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


plt.show()

#save the experiment results

# first, create a dictionary
mdict = {'current_iter': n_iter,
         'logpi_f_trace': logpi_vals.detach().cpu().squeeze().numpy(),
         'running_mean': mean.detach().cpu().squeeze().numpy(),
         'running_var_n': var.detach().cpu().squeeze().numpy(),
         'running_mean_fft': mean_fft.detach().cpu().squeeze().numpy(),
         'running_var_n_fft': var_fft.detach().cpu().squeeze().numpy(),
         'running_mean_abs_fft': mean_abs_fft.detach().cpu().squeeze().numpy(),
         'running_var_abs_fft': var_abs_fft.detach().cpu().squeeze().numpy(),
         'xtrue': img_torch.detach().cpu().squeeze().numpy(),
         'y': img_torch_blurry.detach().cpu().squeeze().numpy(),
         'trace_x': samples.squeeze(),
         'thinning': save_iter,
         'c': cnt,
         'trace_psnr': psnr_vals.cpu().squeeze().numpy(),
         'trace_ssim': ssim_vals.cpu().squeeze().numpy(),
         'sigma2' : sigma2.item(),
         'L': Lip_total.item(),
         'm' : m.item(),
         'h': h.item(),
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

mat_name = exp_name + time_str + ".mat"
# save
scipy.io.savemat(res_path + mat_name, 
                  mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')


