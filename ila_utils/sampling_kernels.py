import torch
import math
import numpy as np

# Defining sampling kernels for each algorithm (ULA, I(M)LA, SKROCK)

def ula_one_step(X: torch.Tensor, Z_k: torch.Tensor, delta, grad_logpi):
    return (X - delta * grad_logpi(X)).detach().clone() + math.sqrt(2*delta) * Z_k

def ila_one_step(X_k: torch.Tensor, Z_k: torch.Tensor, 
                 theta : torch.Tensor, h: torch.Tensor, logpi,
                 tolerance, hist_size, device, verbose=0):

        # optimization settings
        n_iter_max = 1

        # initial value
        x = X_k.to(device).detach().clone()
        x.requires_grad = True

        
        # set up lbfgs optimizer
        optimizer = torch.optim.LBFGS([x], lr=1, max_iter=500, 
                    max_eval=None, tolerance_grad=tolerance,
                    tolerance_change=tolerance,
                    history_size=hist_size,
                    line_search_fn='strong_wolfe')


        def closure():
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            loss = ((1/theta) * 
                logpi(theta * x + (1-theta)*X_k)
                + (1/(2*h))*torch.linalg.matrix_norm(
                    x - X_k - torch.sqrt(2*h) * Z_k, 'fro')**2)
            
            # Backward pass
            loss.backward()

            return loss

        for _ in range(n_iter_max):
            optimizer.step(closure)
            if verbose > 0:
                print(optimizer.state_dict())
        
        return x.detach().clone() # new sample produced by the ILA algorithm


def skrock_one_step(X: torch.Tensor, n_stages, L, eta, delta_perc,
                    grad_logpi):
    # SK-ROCK parameters

    # First kind Chebyshev function

    T_s = lambda s,x: np.cosh(s*np.arccosh(x))

    # First derivative Chebyshev polynomial first kind

    T_prime_s = lambda s,x: s*np.sinh(s*np.arccosh(x))/np.sqrt(x**2 -1)

    # computing SK-ROCK stepsize given a number of stages

    # and parameters needed in the algorithm

    denNStag=(2-(4/3)*eta)

    rhoSKROCK = ((n_stages - 0.5)**2) * denNStag - 1.5 # stiffness ratio

    dtSKROCK = delta_perc*rhoSKROCK/L # step-size

    w0=1 + eta/(n_stages**2) # parameter \omega_0

    w1=T_s(n_stages,w0)/T_prime_s(n_stages,w0) # parameter \omega_1

    mu1 = w1/w0 # parameter \mu_1

    nu1=n_stages*w1/2 # parameter \nu_1

    kappa1=n_stages*(w1/w0) # parameter \kappa_1

    # Sampling the variable X (SKROCK)

    Q=math.sqrt(2*dtSKROCK)*torch.randn_like(X) # diffusion term

    # SKROCK

    # SKROCK first internal iteration (s=1)

    XtsMinus2 = X.clone()

    Xts= X.clone() - mu1*dtSKROCK*grad_logpi(X + nu1*Q) + kappa1*Q

    # s=2,...,n_stages SK-ROCK internal iterations
    for js in range(2,n_stages+1): 

        XprevSMinus2 = Xts.clone()

        mu=2*w1*T_s(js-1,w0)/T_s(js,w0) # parameter \mu_js

        nu=2*w0*T_s(js-1,w0)/T_s(js,w0) # parameter \nu_js

        kappa=1-nu # parameter \kappa_js

        Xts= -mu*dtSKROCK*grad_logpi(Xts) + nu*Xts + kappa*XtsMinus2

        XtsMinus2=XprevSMinus2

    return Xts # new sample produced by the SK-ROCK algorithm



