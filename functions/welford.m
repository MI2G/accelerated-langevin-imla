function [m, S, Sn, g] = welford(x, g, m, S)
%WELFORD computes a running cumulative average (m) and 
% variance (S) and normalized variance (Sn). Supply counting variable g and sample x.
% Sn and m are equivalent to Matlab's mean and var functions.
% initialize algorithm with
% m = X_0 (intitialization of sampling algorithm);
% S = zeros(..);
% g = 0;
% see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.7503&rep=rep1&type=pdf

    g = g + 1;
    Mnext = (g-1)/g * m + (x) / g;
    S = S + (g-1)/g*(x - m).^2;
    m = Mnext;
    if g == 1
        Sn = S / g;
    elseif g > 1
        Sn = S / (g-1);
    end

end
