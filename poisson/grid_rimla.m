
% set desired precision 
precision = [1e-4];

% set desired step size
% precomputed
% corresponding to skrock with [10, 20, 40] stages
step_sizes = [0.000066466585432, 0.000281895425775, 0.001158467948551];

% comparison according to integration time (4.4.1)
%iterations = [80000, 40000, 20000];
%iter_per_sample = [40, 20, 10];
% comparison according to computational cost (4.4.2)
iterations = [80000, 62000, 36000];
iter_per_sample = [40, 31, 18];



N = 2000; %number of samples stored


parfor s = 2:length(step_sizes)
    for p = 1:length(precision)
        poisson_deblurring_rimla_fun(iterations(s), N,...
            iter_per_sample(s), step_sizes(s), precision(p))
    end
end

exit;