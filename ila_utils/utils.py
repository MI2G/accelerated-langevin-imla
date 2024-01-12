
import torch
import hdf5storage
import numpy as np
from PIL import Image, ImageOps

import sampling_tools as st
import cvx_nn_models   

# Defining the blurring operators for the deblurring inverse problems as well as
# handle functions to calculate A*x in the Fourier domain.

# Inspired by

# AUTHORS: Jizhou Li, Florian Luisier and Thierry Blu

# GitHub account : https://github.com/hijizhou/PureLetDeconv

# REFERENCES:
#     [1] J. Li, F. Luisier and T. Blu, PURE-LET image deconvolution, 
#         IEEE Trans. Image Process., vol. 27, no. 1, pp. 92-105, 2018.
#     [2] J. Li, F. Luisier and T. Blu, Deconvolution of Poissonian images with the PURE-LET approach, 
#         2016 23rd Proc. IEEE Int. Conf. on Image Processing (ICIP 2016), Phoenix, Arizona, USA, 2016, pp.2708-2712.
#     [3] J. Li, F. Luisier and T. Blu, PURE-LET deconvolution of 3D fluorescence microscopy images, 
#         2017 14th Proc. IEEE Int. Symp. Biomed. Imaging (ISBI 2017), Melbourne, Australia, 2017, pp. 723-727.

# Adapted in pytorch by: MI2G

# choose an index from 0 to 7 to select a motion blur 
# kernel from levin09 

def motion_blur_operators(size, device, index=0):

    nx = size[0]
    ny = size[1]

    # embed kernel in image of original image size
    h = torch.zeros(nx,ny).to(device)
    # load blur kernels
    blur_dict = hdf5storage.loadmat("../kernels/Levin09.mat")
    # set second index between 0 to 7 to choose between blur kernels
    ker = blur_dict["kernels"][0][index]
    
    # compute indexes for central placement of blur kernel
    size_k = ker.shape[0]
    ind_x = np.arange(np.floor(nx/2-size_k/2), np.ceil(nx/2+size_k/2), dtype=np.int32)
    ind_y = np.arange(np.floor(ny/2-size_k/2), np.ceil(ny/2+size_k/2), dtype=np.int32)

    # place the blur kernel
    h[ind_x[0]:ind_x[-1],ind_y[0]:ind_x[-1]] = torch.from_numpy(ker)

    c = np.ceil(np.array([nx,ny])/2).astype("int64") 

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1)))
    HC_FFT = torch.conj(H_FFT)

    # A forward operator
    A = lambda x: torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)
    # A backward operator
    AT = lambda x: torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x[0,0]))).real.reshape(x.shape)

    AAT_norm = st.max_eigenval(A, AT, nx, 1e-4, int(1e4), 0, device)

    HC_H_FFT = torch.abs(HC_FFT * H_FFT)
    min_ev = torch.sqrt(torch.min(torch.min(HC_H_FFT)))
    max_ev = torch.sqrt(torch.max(torch.max(HC_H_FFT)))

    return A, AT, AAT_norm, min_ev, max_ev

# Defining the convex neural network

# Adapted from https://github.com/axgoujon/convex_ridge_regularizers

def load_crrnn_and_init(sigma_training, t, device):

    sigma_training = 5
    t = 5
    exp_name = f'Sigma_{sigma_training}_t_{t}'
    model = cvx_nn_models.utils.load_model(exp_name, device)

    print(f'Numbers of parameters before prunning: {model.num_params}')
    model.prune()
    #model.prune(change_splines_to_clip=True)
    print(f'Numbers of parameters after prunning: {model.num_params}')

    # [not required] intialize the eigen vector of dimension (size, size) associated to the largest eigen value
    model.initializeEigen(size=100)
    # compute bound via a power iteration which couples the activations and the convolutions
    model.precise_lipschitz_bound(n_iter=100)
    # the bound is stored in the model
    L = model.L.data.item()
    print(f"Lipschitz bound {L:.3f}")


    im = torch.empty((1, 1, 100, 100), device=device).uniform_()
    model.grad(im)# alias for model.forward(im) and hence model(im)

    im = torch.empty((1, 1, 100, 100), device=device).uniform_()
    model.cost(100*im)

    return model, L

# Load images utils

def load_lizard_image(device):
    img = Image.open("../images/87046.jpg")
    img = np.array(ImageOps.grayscale(img))
    img = img[:-1, 60:-101]
    img_torch = torch.tensor(img, device=device, dtype=torch.double).reshape((1,1) + img.shape)/255

    return img, img_torch

def load_castle_image(device):
    img = Image.open("../images/102061.jpg")
    img = np.array(ImageOps.grayscale(img))
    img = img[20:-141,:-1]
    img_torch = torch.tensor(img, device=device, dtype=torch.double).reshape((1,1) + img.shape)/255

    return img, img_torch

def load_person_image(device):
    img = Image.open("../images/101087.jpg")
    img = np.array(ImageOps.grayscale(img))
    img = img[70:-91,:-1]
    img_torch = torch.tensor(img, device=device, dtype=torch.double).reshape((1,1) + img.shape)/255

    return img, img_torch