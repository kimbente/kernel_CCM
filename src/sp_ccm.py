import torch
import numpy as np

# import embedding code from sibling file
from .embedding import embed
from .utils import generate_mask_tensor
from .iaaft import surrogates

device = 'cpu'

###################################
### Simplex Projection (SP) CCM ###
###################################

######################
### differentiable ###
######################

def run_SP_CCM(y, x, filter, max_offset, L, device):
    """ Run CCM as in van Nes and Sugihara using the Simplex algorithm. The tensor containing crossmapping skills for one library size (L) and each configuration of that library size are returned.

    Args:
        y (torch.tensor): Causal Y time series. Will be embedded.
        x (torch.tensor): Causal X time series. Will be predicted.
        filter (torch.Size([K])): binary filter that contains E ones.
        max_offset (torch.Size([1])): maximum positive offset of filter
        L (int): library size
    Returns:
        rho_L_l (torch.Size(1, N])): Pearson's r correlation cofficient between predicted and true. between 0 and 1 where 1 is best.
    """
    large_value = torch.tensor([9999.], device = device)
    minus_one = torch.tensor([-1], device = device)

    #################
    ### Embedding ###
    #################

    # Embed once for the given filter
    y_embeddings, x_gt = embed(filter, y, x, max_offset, device)

    # Extract N (rows) and E (columns) as we use it often: effective dimensionality of data
    N = y_embeddings.shape[0]
    E = y_embeddings.shape[1]

    ##########################
    ### Euclidean distance ###
    ##########################

    # Pairwise distances: y_embeddings are torch.Size([N, E])
    # results in shape torch.Size([N, N]) and symmetric
    dist_euc = torch.cdist(y_embeddings, y_embeddings, p = 2).to(device)

    # Mask distance to self with high value
    # Add large value to diagonal (distance to self) so it effectively never gets used
    dist_euc = dist_euc + (torch.eye(N).to(device) * large_value)

    ############################################
    ### N Library variations l for a given L ###
    ############################################

    # Placeholder rho tensor where columns are l element L (library realisations)
    # we keep all these to look at distribution
    rho_l = torch.empty(size = (0, N)).to(device)

    # Create masks for all possible libraries of length L
    # rows of l_train_masks are all l i.e. N library filters/masks
    # ATTENTION: must we float for multiplication
    l_train_masks = generate_mask_tensor(N, L).int().float().to(device)
    
    # Replace with large value (project back to numeric type)
    l_train_masks[l_train_masks == 0] = large_value
    # rows of l_train_masks are all N library filters/masks

    # Mask all distances
    # torch.Size([N, N, N]) where [dist_i (test_case_i), l, mask_vector] (dist_i is the position)
    masked_distances = torch.mul(dist_euc.unsqueeze(-2), l_train_masks)

    #######################
    ### Simplex weights ###
    #######################

    # Minimum b (b = E + 1) distances for each dist_i, l combination
    # top_Eplus1_indices are absolute indices as nan's would not be top
    # sizes are each torch.Size([N, N, E])
    top_Eplus1_values, top_Eplus1_indices = masked_distances.topk(k = (E + 1), dim = -1, largest = False)
    # torch.Size([N, N, 1])
    top1_values = top_Eplus1_values[:, :, 0]
    # divide all top E distances by the top 1
    ratio_to_top1 = torch.div(top_Eplus1_values, top1_values.unsqueeze(-1))

    # element-wise cac ui with exponential
    # WATCH: possibly a fragile point in computation
    negative_ratios_to_top1 = ratio_to_top1.mul(minus_one)
    ui = torch.exp(negative_ratios_to_top1)
    # This line is needed to ensure differentiability in torch.exp() https://discuss.pytorch.org/t/torch-exp-is-modified-by-an-inplace-operation/90216
    ui = ui + 0

    # Add line for stability: for large ratio_to_top1 ui will go towards 0
    ui[ui < 0.000001] = 0.000001
    # calculate normalising rowsum of ui values
    ui_rowsum = torch.sum(ui, dim = -1)
    # weights
    wi = torch.div(ui, ui_rowsum.unsqueeze(-1))

    ###################
    ### Predictions ###
    ###################

    # indices are absolute (relating to size N)
    # sum over E weighted corresponding x_gt values
    x_preds = torch.sum(torch.mul(x_gt[top_Eplus1_indices], wi), dim = -1)

    ##########################
    ### Crossmap skill rho ###
    ##########################

    # No batchwise corrcoeff implementation so we need to loop through all Libraries
    rho_l = torch.empty(size = (0,)).to(device)

    for li in range(x_preds.shape[- 1]):
        # column of prediction: library view: 
        rho_l = torch.concat((rho_l, torch.corrcoef(torch.vstack((x_gt, x_preds[:, li])))[0, 1].unsqueeze(0)),
                                  dim = 0)

    return rho_l


def SP_CCM_iaaft(y, x_gt_iaaft_selected, selected_train_mask, ccmfilter, max_offset, device):
    # run only for one mas

    y_embeddings, x_gt = embed(ccmfilter, y, x_gt_iaaft_selected, max_offset, device)

    # Extract N (rows) and E (columns) as we use it often: effective dimensionality of data
    N = y_embeddings.shape[0]
    E = y_embeddings.shape[1]

    # truncate mask
    selected_train_mask = selected_train_mask[0: N]
    
    large_value = torch.tensor([9999.], device = device)
    minus_one = torch.tensor([-1], device = device)

    # Pairwise distances: y_embeddings are torch.Size([N, E])
    # results in shape torch.Size([N, N]) and symmetric
    dist_euc = torch.cdist(y_embeddings, y_embeddings, p = 2).to(device)

    # Mask distance to self with high value
    # Add large value to diagonal (distance to self) so it effectively never gets used
    dist_euc = dist_euc + (torch.eye(N).to(device) * large_value)

    selected_train_mask = selected_train_mask.int().float().to(device)
    selected_train_mask[selected_train_mask == 0.] = large_value

    masked_distances = torch.mul(dist_euc, selected_train_mask)

    #######################
    ### Simplex weights ###
    #######################

    # Minimum b (b = E + 1) distances for each dist_i, l combination
    # top_Eplus1_indices are absolute indices as nan's would not be top
    # sizes are each torch.Size([N, E])
    top_Eplus1_values, top_Eplus1_indices = masked_distances.topk(k = (E + 1), dim = -1, largest = False)

    # torch.Size([N, 1])
    top1_values = top_Eplus1_values[:, 0]

    # torch.Size([N, 1])
    ratio_to_top1 = torch.div(top_Eplus1_values, top1_values.unsqueeze(-1))

    # element-wise cac ui with exponential
    # WATCH: possibly a fragile point in computation
    negative_ratios_to_top1 = ratio_to_top1.mul(minus_one)
    ui = torch.exp(negative_ratios_to_top1)
    # This line is needed to ensure differentiability in torch.exp() https://discuss.pytorch.org/t/torch-exp-is-modified-by-an-inplace-operation/90216
    ui = ui + 0


    # Add line for stability: for large ratio_to_top1 ui will go towards 0
    ui[ui < 0.000001] = 0.000001
    # calculate normalising rowsum of ui values
    ui_rowsum = torch.sum(ui, dim = -1)
    # weights
    wi = torch.div(ui, ui_rowsum.unsqueeze(-1))

    ###################
    ### Predictions ###
    ###################

    # indices are absolute (relating to size N)
    # sum over E weighted corresponding x_gt values
    x_preds = torch.sum(torch.mul(x_gt[top_Eplus1_indices], wi), dim = -1)

    rho = torch.corrcoef(torch.vstack((x_gt, x_preds)))[0, 1]

    return rho


def run_ccm_experiment(causal_x, causal_y, ccm_filter, ccm_shift, n_train, device):
    
    result_status = True

    noise_increment = torch.tensor([0.025], device = device)
    noise_creeper = torch.tensor([0.], device = device) - noise_increment

    # True: we have nan
    while (result_status == True):

        # increment noise (0 in first iteration)
        noise_creeper = noise_creeper + noise_increment

        # Add noise (0 in first iteration)
        causal_y = causal_y + (torch.randn(causal_y.shape[0], device = device) * noise_creeper)
        
        ###########
        ### CCM ###
        ###########

        rho_l = run_SP_CCM(
                        y = causal_y,
                        x = causal_x,
                        filter = ccm_filter,
                        max_offset = ccm_shift,
                        L = n_train,
                        device = device)
        
        #######################
        ### CCM INDEPENDENT ###
        #######################

        # Generate N surrogate (permuted ts)
        causal_y_N_iaaft = torch.tensor(surrogates(x = causal_y.cpu(), ns = causal_y.shape[0], tol_pc = 5., verbose = False), dtype = torch.float32)

        l_train_masks = generate_mask_tensor(causal_y.shape[0], n_train)

        # placeholder
        rho_l_ind = torch.empty(size = (1, 0)).to(device)

        # N passes with a different surrogate each time
        for l in range(causal_y_N_iaaft.shape[0]):

            rho_ind = SP_CCM_iaaft(y = causal_y, 
                            x_gt_iaaft_selected = causal_y_N_iaaft[l], 
                            selected_train_mask = l_train_masks[l], 
                            max_offset = ccm_shift,
                            ccmfilter = ccm_filter, 
                            device = device)
            
            rho_l_ind = torch.cat((rho_l_ind, rho_ind.unsqueeze(0).unsqueeze(0)), dim = 1)

        p95 = torch.tensor([0.95]).to(device)

        result_status = (rho_l.isnan().any() & rho_l_ind.isnan().any())

        if (result_status == True):
            print("We get nan's and have to increase the noise level.")

    # print once while loop is finished
    print("Rho mean", np.round(rho_l.mean().item(), 3))
    print("Rho std", np.round(rho_l.std().item(), 3))
    print("Rho indep. p95", np.round(torch.quantile(rho_l_ind, p95).item(), 3))

    print("Added noise", np.round(noise_creeper.item(), 3))

    return(rho_l.mean(), rho_l.std(), torch.quantile(rho_l_ind, p95), noise_creeper)