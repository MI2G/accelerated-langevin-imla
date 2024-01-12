function [X, grad_pi, log_pi, stepsize, alpha, flag] = mala_update(Xk, ...
    gradf, logpi_f, ...
    grad_xk, log_pi_x, stepsize, max_step, k, alpha_star)
%MALA using METROPOLIS_HASTINGS step
%

% current X_k is set, generate Brownian motion
Z_k = randn(size(Xk));

% compute proposal X_kp1
X_kp1 = Xk(:) - stepsize * gradf(Xk(:)) + sqrt(2*stepsize) * Z_k(:);
log_pi_proposal = -logpi_f(X_kp1);
% we know that from before
% log_pi_x = -logpi_f(X_k(:));
grad_proposal = gradf(X_kp1);
% we know that from before
%grad_xk = gradf(X_k(:));

q = @ (x_prime, x, grad_pi) (-1/(4*stepsize) ...
    * norm(x_prime - (x - stepsize * grad_pi), 2)^2);

rnd_number = log(rand);

alpha = min(0, log_pi_proposal + q (Xk, X_kp1, grad_proposal) - (log_pi_x + q(X_kp1, Xk, grad_xk)));

if rnd_number < alpha %%% accept
    X = X_kp1;
    grad_pi = grad_proposal;
    log_pi = log_pi_proposal;
    flag = 1;
    %sprintf("accept")
else %%% reject
    X = Xk;
    grad_pi = grad_xk;
    log_pi = log_pi_x;
    flag = 0;
    %sprintf("reject")
end

% adjust stepsize
gamma = 10*max_step * k .^ (-0.8);
stepsize = stepsize + gamma * (alpha - log(alpha_star));

%debug
%if mod(k,1000) == 0
%    stepsize
%end

stepsize = clip(stepsize, 0, max_step);

% xvals = linspace(1, 1000000, 1000000);
% figure()
% plot(xvals, gamma(xvals))
% hold on 

end

function y = clip(x,bl,bu)
  % return bounded value clipped between bl and bu
  y=min(max(x,bl),bu);

end

