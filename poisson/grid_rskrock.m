
% set desired stages vector

stages = [10, 20, 40];

iterations = [80000, 40000, 20000];
thinning = [40, 20, 10];

N = 2000;

parfor  p = 1:length(stages)
    poisson_deblurring_TV_rskrock_fun(iterations(p), N, stages(p), thinning(p))
end

exit;