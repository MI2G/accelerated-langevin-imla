% Numerical experiment
%
% Poisson Deblurring problem with TV
% 
% R-IMLA
%
% Written as a function that takes step_size and desired precision
%
% This function will run for iter_sampling iterations,
% collect a trace for N samples, saves one image to the trace each
% iter_per_sample, using given step_size and precision.
%
% Teresa Klatzer

function out = poisson_deblurring_rimla_fun(iter_sampling, N, iter_per_sample, ...
    step_size, precision)

assert(iter_sampling/iter_per_sample == N)

addpath("../libs/L-BFGS-B-C/Matlab/");
addpath("../functions/")

h = step_size;
saving_frequency = iter_sampling / 10;
ex_name = sprintf("poiss_deblurTV_cman_rimla_h_%.2e_precision_%.1e", h, precision);

date_str = datetime("today");

first_write = 1;

%%% initialize the random number generator to make the results repeatable
rng('default');
%%% initialize the generator using a seed of 1
rng(1);

%% Setup experiment
% load image in range [0...1] 
x = double(importdata("../images/cameraman.txt"))/255.0;

% choose MIV and then calculate Photon level
mean_x = mean(mean(x));
%hyperparameters
pref_image_mean = 10; %MIV
eta = pref_image_mean / mean_x;

beta_tv = 0.6 * eta;
beta_poiss = 0.01 * pref_image_mean;

xtrue = eta * x;

[nRows, nColumns] = size(xtrue); % size of the image
image_sz = [nRows, nColumns];

%%% operators A and A' operating with column vectors
[A, AT, ~, H_FFT, HC_FFT] = box_blur(image_sz, 5, 1);

% generate 'y'
y = poissrnd(A(xtrue));

%show observation
figure(2)
imagesc(reshape(y,image_sz));
colormap gray
colorbar;

sprintf("PSNR (noisy): %.2f", psnr(reshape(y, image_sz), xtrue, max(max(xtrue))))

%%% Algorithm parameters & Lipschitz constants
op_norm = norm((HC_FFT(:).*H_FFT(:)),inf); % is 1 in the case of uniform blur
L_fb = eta^2 * (max(max(y))/beta_poiss^2) * op_norm; %% Lips. of the likelihood
lambda = 1/L_fb; %%% regularization parameter for TV approximation
Ltv = 1/lambda; %%%  Lipschitz constant of the prior TV
L = L_fb + Ltv; %%% Lipschitz constant of the model 
theta = 0.5;

% Gradients, proximal and \log\pi trace generator function using column
% vectors
proxTV = @(x) reshape(chambolle_prox_TV_stop(reshape(x, image_sz), ...
    'lambda',beta_tv*lambda,'maxiter',25), [],1);
%%% gradient of the likelihood see bottom, here's the function handle
gradLike = @(x) gradLike_fun(x, eta, beta_poiss, A, AT,y);
gradTV = @(x) (x - proxTV(reshape(x, image_sz)))/lambda; %%% gradient of the prior TV

gradf = @(x) gradLike(x) +  gradTV(x); %%% gradient of the model
like_fb = @(x) sum(A(eta*x) + beta_poiss - log(A(eta*x) + beta_poiss) .* y);
logpi_f = @(x) like_fb(x) ...
    + beta_tv*reshape(TVnorm(reshape(x, image_sz)),[],1); %%% logpi likelihood & TV prior

%initialize 
X_k = y./eta;

%%% solve with lbfgsb

% options
opts    = struct( 'x0', X_k(:) );
opts.printEvery     = 100;
opts.m  = 5;

% There are no constraints
l   = -inf(length(X_k(:)),1);
u   = inf(length(X_k(:)),1);

% Ask for given accuracy
opts.pgtol      = precision;
opts.factr      = precision/eps; 
opts.maxIts     = 1000;


%%% wrap around sampling

trace_1px = zeros(iter_sampling, 1);
logpi_f_trace = zeros(iter_sampling, 1);
trace_x = zeros(N, nRows * nColumns);

info = cell(iter_sampling, 1);
trace_mean_x = zeros(N, nRows * nColumns);
trace_var_x = zeros(N, nRows * nColumns);
trace_psnr = zeros(N,1);
trace_nmrse = zeros(N,1);
logpi_f_trace_thinned = zeros(N,1);

running_mean = abs(X_k(:));
running_var = zeros(size(running_mean));
running_var_n = zeros(size(running_mean));
r_counter = 0;
c = 0;

%%% prepare folder for experiment results
result_dir = sprintf("results/%s_%s_iter%d", ex_name, date_str, iter_sampling)
if ~exist(result_dir, 'dir')
       mkdir(result_dir)
end

%%% start the diary
fname_diary = sprintf("%s/diary.txt", result_dir );
diary(fname_diary)

%%% continue filename preparations
fname = sprintf('%s/data.mat', result_dir)
fname_artifacts = sprintf("%s/", result_dir)

fname_fig0 = "noisy.png";
saveas(gcf,sprintf('%s/%s', fname_artifacts, fname_fig0))

overall_timer = tic;

for k = 1:iter_sampling

    % compute running mean and var
    [running_mean, running_var, running_var_n, r_counter] = welford(X_k(:), r_counter, running_mean, running_var);

    % set x0 new every sampling iteration
    opts.x0 = X_k(:);

    % current X_k is set, generate Brownian motion
    Z_k = randn(nRows, nColumns);

    % define function handles with new X_k, Z_k
    gradF = @(x) gradf(theta * x + (1-theta)*X_k(:)) + (1/h) * (x - X_k(:) - sqrt(2*h)*Z_k(:));
    logpi_F = @(x) (1/theta) * logpi_f(theta * x + (1-theta)*X_k(:)) + (1/(2*h))*norm(x-X_k(:)-sqrt(2*h)*Z_k(:), 'fro')^2;
    % set "errFcn" with new gradF
    opts.errFcn     = @(x) norm(gradF(x), 2);
    % "outputFcn" will save values in the "info" output
    opts.outputFcn  = opts.errFcn;

    % optimize and update X_k
    [X_kp1,F_kp1,info_kp1] = lbfgsb( {logpi_F,gradF} , l, u, opts );
    info{k} = info_kp1;
    X_k = abs(X_kp1);

    % record some statistics
    logpi_f_trace(k) = logpi_f(X_k);
    trace_1px(k) = X_k(333);

    if mod(k, 1000) == 0
        fprintf('%d\n', k)
    end


    %record thinned chain
    if mod(k, iter_per_sample) == 0
        c = c + 1;
        trace_x(c,:) = X_k;
        trace_mean_x(c,:) = running_mean;
        trace_var_x(c,:) = running_var_n;
        trace_nmrse(c) = sqrt(norm(xtrue(:)/eta-running_mean)./(nColumns*nRows));
        trace_psnr(c) = psnr(xtrue/eta, reshape(running_mean, nRows, nColumns));
        logpi_f_trace_thinned(c) = logpi_f_trace(k);

    end

    if mod(k,saving_frequency) == 0
        current_iter = k;
        %save res and print something
        fprintf("iteration %d\n", k)
        
        save(fname, '-v7.3')

        %save 2 figures
        figure(1)
        imagesc(reshape(running_mean, image_sz))
        colorbar
        colormap gray
        fname_fig1 = "mean.png";
        saveas(gcf,sprintf('%s/%s', fname_artifacts, fname_fig1))
        figure(2)
        imagesc(reshape(running_var_n, image_sz))
        colorbar
        colormap gray
        fname_fig2 = "var.png";
        saveas(gcf,sprintf('%s/%s', fname_artifacts, fname_fig2))

        %save text file with latest psnr
        fname_txt = "stats.txt";
        if first_write
            fileID = fopen(sprintf('%s/%s', fname_artifacts, fname_txt),'w');
            first_write = 0;
        else
            fileID = fopen(sprintf('%s/%s', fname_artifacts, fname_txt),'a');
        end
        fprintf(fileID,'PSNR: %.03f, current iter: %d, c: %d\n', trace_psnr(c), k , c);
        fprintf(fileID,'NMRSE: %.05f, current iter: %d, c: %d\n', trace_nmrse(c), k , c);
        fclose(fileID);

    end


end

runtime = toc(overall_timer);
runtime_per_iter = runtime/iter_sampling;

save(fname, '-v7.3')
%copy to onedrive
 
diary off

end

function [dx] = gradLike_fun(x, eta, beta, A, AT,y)

    Kx = A(eta*x);
    inv_Kx = 1./(Kx + beta);
    dx = eta * (1 - AT(y .* inv_Kx));

end


