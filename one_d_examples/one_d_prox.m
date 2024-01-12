% one dimensional distributions
% prox map 
% IMLA

close all

set(0,'defaultAxesFontSize', 22)

date_str = datetime('today');
result_dir = sprintf("%s_%s", date_str, "one_d_distributions");
if ~exist(result_dir, 'dir')
       mkdir(result_dir)
end


% set delta
delta = 0.05;
%delta = 1e-4; %for uniform
% set lambda (myula)
lambda = delta;
%set theta
theta = 0.5;
theta1 = 1;

% set type of prox
% 1 ... laplace
% 2 ... uniform
% 3 ... x^4
% 4 ... x^2
% 5 ... cauchy
% 6 ... student-t
prox_type = 1;

n_iter = 15e6;
plot_n = 1e6;
samples_imla = zeros(n_iter, 1);
samples_imla(1) = rand(1,1);
samples_myula = zeros(n_iter, 1);
samples_myula(1) = rand(1,1);
samples_ila = zeros(n_iter, 1);
samples_ila(1) = rand(1,1);

h = waitbar(0,'Please wait...');

if prox_type == 1
    for i = 1:n_iter-1
        waitbar(i / (n_iter-1), h)
        samples_imla(i+1) = 1/theta * prox_map_abs(samples_imla(i) + ...
            theta * sqrt(2*delta)*randn(1,1), delta, theta) ...
            - (1-theta)/theta * samples_imla(i);
        samples_ila(i+1) = 1/theta1 * prox_map_abs(samples_ila(i) + ...
            theta1 * sqrt(2*delta)*randn(1,1), delta, theta1) ...
            - (1-theta1)/theta1 * samples_ila(i);
        samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
            - prox_map_abs(samples_myula(i), 1, lambda)) ...
            + sqrt(2*delta)*randn(1,1);
    end
elseif prox_type == 2
    for i = 1:n_iter-1
        waitbar(i / (n_iter-1), h)
        samples_imla(i+1) = 1/theta * prox_map_unif(samples_imla(i) + ...
            theta * sqrt(2*delta)*randn(1,1)) ...
            - (1-theta)/theta * samples_imla(i);
        samples_ila(i+1) = 1/theta1 * prox_map_unif(samples_ila(i) + ...
            theta1 * sqrt(2*delta)*randn(1,1)) ...
            - (1-theta1)/theta1 * samples_ila(i);
        samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
        - prox_map_unif(samples_myula(i))) ...
        + sqrt(2*delta)*randn(1,1);
    end
elseif prox_type == 3
    for i = 1:n_iter-1
        waitbar(i / (n_iter-1), h)
        samples_imla(i+1) = 1/theta * prox_map_x4(samples_imla(i) + ...
            theta * sqrt(2*delta)*randn(1,1), delta, theta) ...
            - (1-theta)/theta * samples_imla(i);    
        samples_ila(i+1) = 1/theta1 * prox_map_x4(samples_ila(i) + ...
            theta1 * sqrt(2*delta)*randn(1,1), delta, theta1) ...
            - (1-theta1)/theta1 * samples_ila(i); 
        samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
        - prox_map_x4(samples_myula(i), 1, lambda)) ...
        + sqrt(2*delta)*randn(1,1);
    end
elseif prox_type == 4
    for i = 1:n_iter-1
        waitbar(i / (n_iter-1), h)
        samples_imla(i+1) = 1/theta * prox_map_x2(samples_imla(i) + ...
            theta * sqrt(2*delta)*randn(1,1), delta, theta) ...
            - (1-theta)/theta * samples_imla(i);  
        samples_ila(i+1) = 1/theta1 * prox_map_x2(samples_ila(i) + ...
            theta1 * sqrt(2*delta)*randn(1,1), delta, theta1) ...
            - (1-theta1)/theta1 * samples_ila(i);
        samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
        - prox_map_x2(samples_myula(i), 1, lambda)) ...
        + sqrt(2*delta)*randn(1,1);
    end
 elseif prox_type == 5
        for i = 1:n_iter-1
            waitbar(i / (n_iter-1), h)
            samples_imla(i+1) = 1/theta * prox_map_cauchy(samples_imla(i) + ...
                theta * sqrt(2*delta)*randn(1,1), delta, theta) ...
                - (1-theta)/theta * samples_imla(i);  
            samples_ila(i+1) = 1/theta1 * prox_map_cauchy(samples_ila(i) + ...
                theta1 * sqrt(2*delta)*randn(1,1), delta, theta1) ...
                - (1-theta1)/theta1 * samples_ila(i);  
            samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
            - prox_map_cauchy(samples_myula(i), 1, lambda)) ...
            + sqrt(2*delta)*randn(1,1);
        end
elseif prox_type == 6
    for i = 1:n_iter-1
        waitbar(i / (n_iter-1), h)
        samples_imla(i+1) = 1/theta * prox_map_studentt(samples_imla(i) + ...
            theta * sqrt(2*delta)*randn(1,1), delta, theta) ...
            - (1-theta)/theta * samples_imla(i);  
        samples_ila(i+1) = 1/theta1 * prox_map_studentt(samples_ila(i) + ...
            theta1 * sqrt(2*delta)*randn(1,1), delta, theta1) ...
            - (1-theta1)/theta1 * samples_ila(i);  
        samples_myula(i+1) = samples_myula(i) - delta/lambda * (samples_myula(i) ...
        - prox_map_studentt(samples_myula(i), 1, lambda)) ...
        + sqrt(2*delta)*randn(1,1);
    end
end

% Plot traces
figure(2*prox_type)
hold on
plot(samples_myula(1:plot_n), 'DisplayName', "MYULA", 'LineWidth', 2, "Color", "#EDB120")
plot(samples_imla(1:plot_n), 'DisplayName', "IMLA",'LineWidth', 2, "Color", "#0072BD")
plot(samples_ila(1:plot_n), 'DisplayName', "ILA", 'LineWidth', 2, "Color", "#4DBEEE")


legend('Location', 'northwest')

if prox_type == 1
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "laplace", n_iter, delta);
elseif prox_type == 2
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "uniform", n_iter, delta);
elseif prox_type == 3
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "x4", n_iter, delta);
elseif prox_type == 4
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "x2", n_iter, delta);
elseif prox_type == 5
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "cauchy", n_iter, delta);
elseif prox_type == 6
fname_trace = sprintf("%s_%s_%d_delta_%.2e.png", "trace", "studentt", n_iter, delta);
end
% save
saveas(gcf,sprintf('%s/%s', result_dir, fname_trace))
saveas(gcf,sprintf('%s/%s', result_dir, fname_trace_fig))


figure(prox_type*3)
histogram(samples_ila, 'Normalization','pdf', 'EdgeColor','none', 'FaceColor', "#4DBEEE")
hold on 
if prox_type == 1
    plot(linspace(-3,3,1000), 0.5*exp(-abs(linspace(-3,3,1000))), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-3 3])
    ylim([0 0.55])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "laplace", "ila", delta, theta1);
elseif prox_type == 2
    plot(linspace(-2,2,1000), pdf(makedist('Uniform'),linspace(-2,2,1000)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-1 2])
    ylim([0 1.2])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "uniform", "ila", delta, theta1);
elseif prox_type == 3
    plot(linspace(-2,2,1000), 1/(2* gamma(5/4))* exp(-linspace(-2,2,1000).^4), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "x4", "ila", delta, theta1);
    xlim([-2 2])
elseif prox_type == 4
    plot(linspace(-2,2,1000), (1/(sqrt(pi))* exp(-linspace(-2,2,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "x2", "ila", delta, theta1);
elseif prox_type == 5
    plot(linspace(-10,10,1000), 1./(pi*(1+linspace(-10,10,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-10 10])
    ylim([0 0.35])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "cauchy", "ila", delta, theta1);
elseif prox_type == 6
    plot(linspace(-5,5,1000), (gamma(3/2)/(sqrt(2*pi)*gamma(1)))*(1+0.5*linspace(-5,5,1000).^2).^(-3/2), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "studentt", "ila", delta, theta1);
end

saveas(gcf,sprintf('%s/%s', result_dir, fname_fig1))
saveas(gcf,sprintf('%s/%s', result_dir, fname_fig2))

figure(prox_type)
histogram(samples_imla, 'Normalization','pdf', 'EdgeColor','none', 'FaceColor',"#0072BD")
hold on 
if prox_type == 1
    plot(linspace(-3,3,1000), 0.5*exp(-abs(linspace(-3,3,1000))), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-3 3])
    ylim([0 0.55])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "laplace", "imla", delta, theta);
elseif prox_type == 2
    plot(linspace(-2,2,1000), pdf(makedist('Uniform'),linspace(-2,2,1000)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-1 2])
    ylim([0 1.2])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "uniform", "imla", delta, theta);
elseif prox_type == 3
    plot(linspace(-2,2,1000), 1/(2* gamma(5/4))* exp(-linspace(-2,2,1000).^4), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "x4", "imla", delta, theta);
    xlim([-2 2])
elseif prox_type == 4
    plot(linspace(-2,2,1000), (1/(sqrt(pi))* exp(-linspace(-2,2,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "x2", "imla", delta, theta);
elseif prox_type == 5
    plot(linspace(-10,10,1000), 1./(pi*(1+linspace(-10,10,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-10 10])
    ylim([0 0.35])
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "cauchy", "imla", delta, theta);

elseif prox_type == 6
    plot(linspace(-5,5,1000), (gamma(3/2)/(sqrt(2*pi)*gamma(1)))*(1+0.5*linspace(-5,5,1000).^2).^(-3/2), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig1 = sprintf("%s_%s_delta_%.2e_theta%.1f.png", "studentt", "imla", delta, theta);
end

saveas(gcf,sprintf('%s/%s', result_dir, fname_fig1))
saveas(gcf,sprintf('%s/%s', result_dir, fname_fig2))


figure(prox_type * 10)
histogram(samples_myula, 'Normalization','pdf', 'EdgeColor','none', "FaceColor", "#EDB120")
hold on 
if prox_type == 1
    plot(linspace(-3,3,1000), 0.5*exp(-abs(linspace(-3,3,1000))), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-3 3])
    ylim([0 0.55])
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "laplace", "myula", delta);
elseif prox_type == 2
    plot(linspace(-2,2,1000), pdf(makedist('Uniform'),linspace(-2,2,1000)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-1 2])
    ylim([0 1.2])
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "uniform", "myula", delta);
elseif prox_type == 3
    plot(linspace(-2,2,1000), 1/(2* gamma(5/4))* exp(-linspace(-2,2,1000).^4), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "x4", "myula", delta);
    xlim([-2 2])
elseif prox_type == 4
    plot(linspace(-2,2,1000), (1/(sqrt(pi))* exp(-linspace(-2,2,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "x2", "myula", delta);
elseif prox_type == 5
    plot(linspace(-10,10,1000), 1./(pi*(1+linspace(-10,10,1000).^2)), 'k', 'LineStyle','--', 'LineWidth', 2)
    xlim([-10 10])
    ylim([0 0.35])
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "cauchy", "myula", delta);
elseif prox_type == 6
    plot(linspace(-5,5,1000), (gamma(3/2)/(sqrt(2*pi)*gamma(1)))*(1+0.5*linspace(-5,5,1000).^2).^(-3/2), 'k', 'LineStyle','--', 'LineWidth', 2)
    fname_fig2 = sprintf("%s_%s_delta_%.2e.png", "studentt", "myula", delta);
end

saveas(gcf,sprintf('%s/%s', result_dir, fname_fig2))
saveas(gcf,sprintf('%s/%s', result_dir, fname_fig3))



function [x] = prox_map_abs(arg, delta, theta)

    x = wthresh(arg, 's', delta*theta);

end

function [x] = prox_map_unif(arg)

    x = min(1, max(arg, 0));

end

function [x] = prox_map_x4(u, delta, theta)
    lambda = delta* theta;
    x = (u/(8*lambda) + (1/(1728*lambda^3) + u^2/(64*lambda^2))^(1/2))^(1/3) - ...
        1/(12*lambda*(u/(8*lambda) + (1/(1728*lambda^3) + ...
        u^2/(64*lambda^2))^(1/2))^(1/3));
end

function [x] = prox_map_x2(u, delta, theta)
    lambda = delta * theta;
    x = u ./ (2 *lambda + 1);
end

function [y] = prox_map_studentt(x, delta, theta)
    lambda = delta*theta;

    y1 = x/3 + (x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3) - (- x^2/9 + lambda + 2/3)/(x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3);
    y2 = x/3 - (x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3)/2 + (- x^2/9 + lambda + 2/3)/(2*(x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3)) - (3^(1/2)*((x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3) + (- x^2/9 + lambda + 2/3)/(x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3))*1i)/2;
    y3 = x/3 - (x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3)/2 + (- x^2/9 + lambda + 2/3)/(2*(x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3)) + (3^(1/2)*((x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3) + (- x^2/9 + lambda + 2/3)/(x - (x*(3*lambda + 2))/6 + ((- x^2/9 + lambda + 2/3)^3 + (x - (x*(3*lambda + 2))/6 + x^3/27)^2)^(1/2) + x^3/27)^(1/3))*1i)/2;

    y_vec = [y1, y2, y3];
    [val,idx] = sort(imag(y_vec)) ;
    y = y_vec(idx(val==0));
    if isempty(y)
        y = real(y_vec(idx(abs(val)<1e-3)));
    end
 
end


function [y] = prox_map_cauchy(x, delta, theta)
    lambda = delta*theta; 

    y_vec = roots([1, -x, 1+2*lambda, -x]);
    [val,idx] = sort(imag(y_vec)) ;
    y = y_vec(idx(val==0));
    if isempty(y)
        y = real(y_vec(idx(abs(val)<1e-3)));
    end

end







