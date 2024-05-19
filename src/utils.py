import torch

####################
### Library mask ###
####################

def generate_mask_tensor(N, L):
    """Generate masking tensor

    Args:
        N (_type_): Length of embedding
        L (_type_): Length of training data sequence (temporally consecutive)

    Returns:
        mask_tensor (torch.size([N, N])): square tensor which is a boolean (True/False) mask which can be overlain onto the embeddings to generate various training data sets. Rows represent all N different libraries and columns represent the masks for each position.

    Check: 
        Check for one in the bottom right corner and run mask_tensor.sum(dim = 0) (column sum) and mask_tensor.sum(dim = 1) (rowsum)

    Slicing:
        y_embeddings[mask_tensor[0],:]
        x_gt[mask_tensor[0]]
    """
    # create empty tensor with N columns that we add rows to step by step
    mask_tensor = torch.empty(size = (0, N))
    for i in range(N):
        # we loop through i which is the starting position of the consecutive training sequence of length L (L for Library)
        # Case 1: Library does not overspill so the row vector is Falses, Trues and Falses
        if (i + L) < N:
            row = torch.concat((torch.zeros(size = (i, ), dtype = torch.bool),
                                torch.ones(size = (L, ), dtype = torch.bool),
                                torch.zeros(size = (N - L - i, ), dtype = torch.bool)))
            # concat
            mask_tensor = torch.concat((mask_tensor, row.unsqueeze(0)), dim = 0)

        # Case 2: Library does overspill so we have Trues, Falses and Trues
        else:
            other_row = torch.concat((torch.ones(size = (- N + i + L, ), dtype = torch.bool),
                                torch.zeros(size = (N - L, ), dtype = torch.bool),
                                torch.ones(size = (N - i, ), dtype = torch.bool)))
            # concat
            mask_tensor = torch.concat((mask_tensor, other_row.unsqueeze(0)), dim = 0)
        
    return mask_tensor.to(torch.bool)

######################
### FILTER PENALTY ###
######################

def filter_penalty(filter, shift = 0.75, gelu_scalar = 8, l2_scalar = 1):
    # 0.75 is the offset we need

    # L2 regularisation: Sum of squares L2 regularisation for parsimony
    reg_penalty = filter.mul(l2_scalar).square()

    # sigmoid does not assign higher penalty to larger magnitude negative terms. Not best for graidents
    # multiply by scalar
    filter_copy = filter.clone()
    filter_copy = filter_copy.mul(gelu_scalar).add(shift)
    neg_penalty = torch.nn.functional.gelu( - filter_copy)
    
    # Add both terms
    penalty = reg_penalty.add(neg_penalty)
    # Return sum 
    return penalty.sum()