% Numerical experiment
%
% Poisson Deblurring problem with TV
% 
% Reflected SKROCK
%
% Teresa Klatzer

function out = poisson_deblurring_TV_rskrock_fun(iter_sampling, N, n_stages, thinning)

assert(iter_sampling/thinning == N)

addpath("../functions/")


saving_frequency = iter_sampling / 10;
ex_name = sprintf("poiss_deblurTV_cman_rskrock_s_%d", n_stages);


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

figure(1)
imagesc(xtrue)
colormap gray
colorbar

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
L = L_fb + Ltv; %%% Lipschitz constant of the model (simplified: 2*L_fb + Lg)

%% amend ex_name with parameters
ex_name = sprintf("%s_beta_tv%.02f_beta_poiss%.02f_MIV%d", ex_name, beta_tv, beta_poiss, pref_image_mean);

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



%%% wrap around sampling

trace_1px = zeros(iter_sampling, 1);
logpi_f_trace = zeros(iter_sampling, 1);
trace_x = zeros(iter_sampling/thinning, nRows * nColumns);
trace_mean_x = zeros(iter_sampling/thinning, nRows * nColumns);
trace_var_x = zeros(iter_sampling/thinning, nRows * nColumns);
trace_psnr = zeros(iter_sampling/thinning,1);
trace_nmrse = zeros(iter_sampling/thinning,1);
logpi_f_trace_thinned = zeros(iter_sampling/thinning,1);

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

    % update X_k
    X_k = SKROCK_reflected(X_k, L, n_stages, 0.99, gradf);

    % record some statistics
    logpi_f_trace(k) = logpi_f(X_k);
    trace_1px(k) = X_k(333);

    if mod(k, 1000) == 0
        fprintf('%d\n', k)
    end


    %record thinned chain
    if mod(k, thinning) == 0
        c = c + 1;
        trace_x(c,:) = X_k;
        trace_mean_x(c,:) = running_mean(:);
        trace_var_x(c,:) = running_var_n(:);
        trace_nmrse(c) = sqrt(norm(xtrue/eta-reshape(running_mean, image_sz), "fro")./(nColumns*nRows));
        trace_psnr(c) = psnr(xtrue/eta, reshape(running_mean, image_sz), 1.0);
        logpi_f_trace_thinned(c) = logpi_f_trace(k);

    end

    if mod(k,saving_frequency) == 0
        current_iter = k;
        %save res and print something
        fprintf("iteration %d\n", k)
        
        %save results
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
 
diary off

end

function [dx] = gradLike_fun(x, eta, beta, A, AT,y)

    Kx = A(eta*x);
    inv_Kx = 1./(Kx + beta);
    dx = eta * (1 - AT(y .* inv_Kx));

end


