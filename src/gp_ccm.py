import torch
import math # for pi
import sigkernel

from .embedding import embed
from .utils import generate_mask_tensor

device = 'cpu'

#################################
### Gaussian Process (GP) CCM ###
#################################

### Kernel helper function ###

def squared_exponential_kernel(first_embedding, second_embedding, l, sigma):
    """ Calculates squared exponential kernel/covariance between two embedding tensors with name embedding dim E and cardinality N and M. https://www.cs.toronto.edu/~duvenaud/cookbook/.
    Uses torch.cdist() function which calculates the p-norm. The p-norm where p = 2 is equavalent to the Euclidean distance: https://pytorch.org/docs/stable/generated/torch.cdist.html
    https://planetmath.org/vectorpnorm 
    For weighted filter we might have distributive property: l(x1 - x2) l * x1 - l * x2

    Args:
        first_embedding (torch.Size([N, E])): first tensor or embeddings
        second_embedding (torch.Size([M, E])): first tensor or embeddings
        l (torch.Size([])): lengthscale value
        sigma (torch.Size([])): output scalar value

    Returns:
        torch.Size([N, M]): pairwise covariance values
    """
    # potentially use .clamp_min(1e-15) as in Gpytorch implementation
    return torch.cdist(first_embedding, second_embedding, p = 2).square().div(-2 * l.square()).exp().mul(sigma.square())

#################################
### Gaussian Process (GP) CCM ###
#################################

def GP_ccm(y_embeddings_train, y_embeddings_test, x_train, x_test, l, sigma, noise, device):
    """
    Runs GP based ccm given input embedding and return rho and lml to optimize. 
    This version should run on device and should also be optimisable (re avoiding inplace operations and non-leaf operations). We implement a zero mean GP. 

    Args:
        y_embeddings_train (_type_): _description_
        y_embeddings_test (_type_): _description_
        x_train (_type_): _description_
        x_test (_type_): _description_
        l (_type_): _description_
        sigma (_type_): _description_
        noise (_type_): _description_

    Returns:
        rho: _description_
        nlml: negative log marginal likelihood (negative so it can be considered a loss which we are trying to minimise)
    """

    K_train_train = squared_exponential_kernel(y_embeddings_train, 
                                              y_embeddings_train, 
                                              l, 
                                              sigma).to(device)
    
    # rows: test, columns: train (test_train order as this is required in mean function)
    K_test_train = squared_exponential_kernel(y_embeddings_test, 
                                              y_embeddings_train, 
                                              l, 
                                              sigma).to(device)
    
    K_test_test = squared_exponential_kernel(y_embeddings_test, 
                                             y_embeddings_test, 
                                             l, 
                                             sigma).to(device)
    
    # need L later, thus step-wise/separete from chol. inverse
    L = torch.linalg.cholesky(K_train_train.add(torch.eye(K_train_train.shape[0], device = device).mul(noise.square())))

    K_train_train_inv = torch.cholesky_inverse(L)

    # W because these are the weights for the y values
    W = K_test_train.matmul(K_train_train_inv)

    # Rasmussen and Williams - 2006 - Gaussian processes for machine learning - page 34 (16)
    # Zero mean function
    x_test_mean_est = W.matmul(x_train)
    # x_test_covar_est = K_test_test.sub(W.matmul(K_test_train.mT))

    rho = torch.stack((x_test, x_test_mean_est)).corrcoef()[0, 1]

    # Rasmussen and Williams - 2006 - Gaussian processes for machine learning - page 35 (17)
    # Negative log marginal likelihood
    nlml = - x_train.matmul(K_train_train_inv).matmul(x_train).div(-2).add(L.diagonal().log().sum().mul(-1)).add(torch.tensor(math.pi).mul(2).log().mul(x_train.shape[0]).div(-2))

    # Likelihood composition
    # term 1
    # x_train.matmul(K_train_train_inv).matmul(x_train).div(-2)
    # term 2 Cholesky trick for log determinant: extract diagonal from cholesky, log and sum
    # L.diagonal().log().sum().mul(-1)
    # term 3
    # torch.tensor(math.pi).mul_(2).log_().mul_(x_train.shape[0]).div_(-2)

    # delete large tensors to project memory
    del K_train_train, K_test_train, K_test_test, L, K_train_train_inv, W
    del x_test_mean_est

    return rho, nlml


def GP_ccm_sig(y_embeddings_train, y_embeddings_test, x_train, x_test, noise, rbf_sigma, device):
    """
    Runs GP based ccm given input embedding and return rho and lml to optimize. 
    This version should run on device and should also be optimisable (re avoiding inplace operations and non-leaf operations). We implement a zero mean GP. 

    Args:
        y_embeddings_train (_type_): _description_
        y_embeddings_test (_type_): _description_
        x_train (_type_): _description_
        x_test (_type_): _description_
        l (_type_): _description_
        sigma (_type_): _description_
        noise (_type_): _description_

    Returns:
        rho: _description_
        nlml: negative log marginal likelihood (negative so it can be considered a loss which we are trying to minimise)
    """
    # Specify the static kernel (for linear kernel use sigkernel.LinearKernel())
    static_kernel = sigkernel.RBFKernel(sigma = rbf_sigma)
    dyadic_order = 3

    max_batch_size = y_embeddings_train.shape[0] + y_embeddings_test.shape[0]

    # Initialize the corresponding signature kernel
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)


    K_train_train = signature_kernel.compute_Gram(y_embeddings_train, 
                                              y_embeddings_train, 
                                              sym = True, max_batch = max_batch_size).to(device)
    
    # PSD
    # L, V = torch.linalg.eig(K_train_train)
    # clamping or relu only for floats
    # L = torch.nn.functional.relu(L.float()).to(torch.complex64)
    # K_train_train = V @ torch.diag_embed(L) @ torch.linalg.inv(V)
    # enforce symmetry is psd errors
    # K_train_train = (K_train_train + K_train_train.T).div(2).to(torch.float32)
    
    # rows: test, columns: train (test_train order as this is required in mean function)
    K_test_train = signature_kernel.compute_Gram(y_embeddings_test, 
                                              y_embeddings_train, 
                                              sym = True, max_batch = max_batch_size).to(device)
    
    K_test_test = signature_kernel.compute_Gram(y_embeddings_test, 
                                             y_embeddings_test, 
                                             sym = True, max_batch = max_batch_size).to(device)
    
    # need L later, thus step-wise/separete from chol. inverse
    L = torch.linalg.cholesky(K_train_train.add(torch.eye(K_train_train.shape[0], device = device).mul(noise.square())))

    K_train_train_inv = torch.cholesky_inverse(L)

    # W because these are the weights for the y values
    W = K_test_train.matmul(K_train_train_inv)

    # Rasmussen and Williams - 2006 - Gaussian processes for machine learning - page 34 (16)
    # Zero mean function
    x_test_mean_est = W.matmul(x_train)
    # x_test_covar_est = K_test_test.sub(W.matmul(K_test_train.mT))

    rho = torch.stack((x_test, x_test_mean_est)).corrcoef()[0, 1]

    # Rasmussen and Williams - 2006 - Gaussian processes for machine learning - page 35 (17)
    # Negative log marginal likelihood
    nlml = - x_train.matmul(K_train_train_inv).matmul(x_train).div(-2).add(L.diagonal().log().sum().mul(-1)).add(torch.tensor(math.pi).mul(2).log().mul(x_train.shape[0]).div(-2))

    # Likelihood composition
    # term 1
    # x_train.matmul(K_train_train_inv).matmul(x_train).div(-2)
    # term 2 Cholesky trick for log determinant: extract diagonal from cholesky, log and sum
    # L.diagonal().log().sum().mul(-1)
    # term 3
    # torch.tensor(math.pi).mul_(2).log_().mul_(x_train.shape[0]).div_(-2)

    # delete large tensors to project memory
    del K_train_train, K_test_train, K_test_test, L, K_train_train_inv, W
    del x_test_mean_est

    return rho, nlml


def run_sigGPCCM_experiment(causal_x, causal_y, sig_filter, sig_shift, rbf_scale, noise_scale, n_train, device):

    y_embeddings, x_gt = embed(filter = sig_filter, 
                               y = causal_y, 
                               x = causal_x,
                               max_pos_offset = sig_shift, 
                               device = device)
    
    N = y_embeddings.shape[0]
    E = y_embeddings.shape[1]

    l_train_masks = generate_mask_tensor(N, n_train)

    # Generate N surrogate (permuted ts) of x_gt
    causal_x_N_iaaft = torch.tensor(surrogates(x = x_gt.cpu(), ns = N, tol_pc = 5., verbose = False), dtype = torch.float32)

    # placeholders
    rho_l = torch.empty(size = (1, 0)).to(device)
    rho_l_ind = torch.empty(size = (1, 0)).to(device)

    for l in range(N):

        ##############
        ### GP-CCM ###
        ##############

        rho, _ = GP_ccm_sig(
                y_embeddings_train = y_embeddings[l_train_masks[l]].unsqueeze(-1).to(device),
                y_embeddings_test = y_embeddings[ ~ l_train_masks[l]].unsqueeze(-1).to(device),
                x_train = x_gt[l_train_masks[l]].to(device),
                x_test = x_gt[ ~ l_train_masks[l]].to(device),
                noise = noise_scale, # for stability
                rbf_sigma = rbf_scale,
                device = device)
        
        ##########################
        ### GP-CCM INDEPENDENT ###
        ##########################

        rho_ind, _= GP_ccm_sig(
                y_embeddings_train = y_embeddings[l_train_masks[l]].unsqueeze(-1).to(device),
                y_embeddings_test = y_embeddings[ ~ l_train_masks[l]].unsqueeze(-1).to(device),
                x_train = causal_x_N_iaaft[l, l_train_masks[l]].to(device),
                x_test = causal_x_N_iaaft[l, ~ l_train_masks[l]].to(device),
                noise = noise_scale, # for stability
                rbf_sigma = rbf_scale,
                device = device)
        
        rho_l = torch.concat((rho_l, rho.unsqueeze(0).unsqueeze(0)), dim = 1)
        rho_l_ind = torch.concat((rho_l_ind, rho_ind.unsqueeze(0).unsqueeze(0)), dim = 1)

    p95 = torch.tensor([0.95]).to(device)

    # print
    print("Rho mean", np.round(rho_l.mean().item(), 3))
    print("Rho std", np.round(rho_l.std().item(), 3))
    print("Rho indep. p95", np.round(torch.quantile(rho_l_ind, p95).item(), 3))

    return(rho_l.mean(), rho_l.std(), torch.quantile(rho_l_ind, p95))
