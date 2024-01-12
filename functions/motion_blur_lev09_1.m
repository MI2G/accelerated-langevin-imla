function [A, AT, ATA, H_FFT, HC_FFT, m_const, max_ev] = motion_blur_lev09_1(sizeX, makeColumn)
%MOTION_BLUR_LEV09_1 
% this function loads motion blur number 1
% from the data set by Levin 2009
% and creates the operators operating on column vectors
% as needed for the LBFGS-B optimizer if makeColumn is
% set to 1
%
% Teresa Klatzer Dec 2022

%%% load the blur
load("../kernels/Levin09.mat")
kernel = kernels{1};

% embed kernel in image of original image size
size_k = length(kernel);

% compute indexes for central placement of blur kernel
ind_x = 1+sizeX(1)/2-(size_k-1)/2 : 1+sizeX(1)/2+(size_k-1)/2;
ind_y = 1+sizeX(2)/2-(size_k-1)/2 : 1+sizeX(2)/2+(size_k-1)/2;
kernelimage = zeros(sizeX);
kernelimage(ind_x,ind_y) = kernel;

%%% operators A and A'
%compute fft kernel
H_FFT = fft2(fftshift(kernelimage));
HC_FFT = conj(H_FFT);

HC_H_FFT = HC_FFT.*H_FFT;
m_const = sqrt(min(min(HC_H_FFT)));
max_ev = sqrt(max(max(HC_H_FFT)));
sprintf("smallest EV: %d", m_const)
sprintf("largest EV: %d", max_ev)


if makeColumn
    %%% operators A and A' operating with column vectors
    A = @(x) reshape(real(ifft2(H_FFT.*fft2(reshape(x, sizeX)))), [],1); % A operator
    AT = @(x) reshape(real(ifft2(HC_FFT.*fft2((reshape(x, sizeX))))), [],1); % A transpose operator
    ATA = @(x) reshape(real(ifft2((HC_FFT.*H_FFT).*fft2((reshape(x, sizeX))))), [],1); % AtA operator
else
    %%% operators A and A' operating with 2D matrix
    A = @(x) real(ifft2(H_FFT.*fft2(reshape(x, sizeX)))); % A operator
    AT = @(x) real(ifft2(HC_FFT.*fft2((reshape(x, sizeX))))); % A transpose operator
    ATA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((reshape(x, sizeX))))); % AtA operator
end

end

