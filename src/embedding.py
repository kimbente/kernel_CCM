import torch

# TODO device doesn't need to be defined locally
# device = "cpu"

#################
### EMBEDDING ###
#################

def embed(filter, y, x, max_pos_offset, device):
    """ We are lag-embedding effect Y to later predict cause X. Similarity is defined in the Y manifold, i.e. the embedding space. We also extract corresponding gt X values. This code is designed to work with binary and weighted filters alike. 
    TODO: For loop can probably be avoided to make code faster but embedding is only run once.

    Args:
        filter (k, 0): filter consisting of (a: binary case) 0 and 1 (b: weighted case) weights
        y (N, ): e.g. 401 length timeseries (to be embedded)
        x (N,):  e.g. 401 length timeseries (target)
        max_pos_offset (1,): integer of max_pos_offset

    Returns:
        y_embeddings (M = N - k + 1, E): embedding y into Y x E dimesions by lagging
        x_gt (M = N - k + 1, ): ground truth x values corresponding to each embedding
        E (1): integer of effective embedding dimensions
    """
    # Span / length of the filter including zeros: We expect the number of (complete) embeddings to be d - k + 1 (because if d = k, say 12 = 12, we still have 1 valid sequence)
    k = torch.tensor(filter.shape[0]).to(device)
    # Extract effective embedding dim (how many ones/non-zero positions)
    # Make inequality so that code works for negative values which we allow during the optimisation process but we discourage through the design of the loss function
    E = torch.tensor(filter[filter != 0].shape[0]).to(device)

    # Case 1
    if (max_pos_offset + 1) > k:
        # print("case 1")
        # if (max_pos_offset = k): one left of embedding filter
        # if (max_pos_offset + 1) == k: space same starting position
        # Y preceedes vector
        y_embeddings = torch.empty(size = (0, E)).to(device)

        for i in range(y.shape[0] - max_pos_offset):
            y_window = y[(i + (max_pos_offset + 1) - k) : (i + (max_pos_offset + 1))].to(device)
            y_window_filtered = torch.mul(y_window, filter).to(device)
            nonzero_filter_indices = torch.nonzero(filter).to(device)
            y_embeddings = torch.concat((y_embeddings, y_window_filtered[nonzero_filter_indices].reshape(1, E)), dim = 0).to(device)
        
        x_gt = x[0:(y.shape[0] - max_pos_offset)].to(device) # pythonic: select first few


    # Case 2
    elif(max_pos_offset < k) & (max_pos_offset > -1):
        # print("case 2")
        # between 0 and k-1

        ### Y EMBEDDINGS ###
        # Intialise empty tensor with the according dim E
        y_embeddings = torch.empty(size = (0, E)).to(device)

        # Plus one because stop in range is excluded
        for i in range(y.shape[0] - k + 1):
            # subset corresponding window (with span k) from y timeseries
            y_window = y[i : (i + k)].to(device)
            # element-wise multiplication
            y_window_filtered = torch.mul(y_window, filter).to(device)
            # extract non-zero elements
            # TODO This step could be weakness if true values are zero. Change implementation if needed
            nonzero_filter_indices = torch.nonzero(filter).to(device)
            # change order of dims from [E, 1] to [1, E] for concatination along first dim
            y_embeddings = torch.concat((y_embeddings, y_window_filtered[nonzero_filter_indices].reshape(1, E)), dim = 0).to(device)

        ### X GROUND TRUTH ###
        # Select the latter x values as ground truth as we have full "history" for these
        # Now goes further back but doesn't reach end
        if max_pos_offset == 0:
            x_gt = x[ - y_embeddings.shape[0]:].to(device)
        else:
            x_gt = (x[ - (y_embeddings.shape[0] + max_pos_offset): - max_pos_offset]).to(device)

    elif(max_pos_offset < 0):
        # print("case 3")

        y_embeddings = torch.empty(size = (0, E)).to(device)

        for i in range(y.shape[0] - k + 1 + max_pos_offset):
            # subset corresponding window (with span k) from y timeseries
            y_window = y[i : (i + k)].to(device)
            # element-wise multiplication
            y_window_filtered = torch.mul(y_window, filter).to(device)
            # extract non-zero elements
            # TODO This step could be weakness if true values are zero. Change implementation if needed
            nonzero_filter_indices = torch.nonzero(filter).to(device)
            # change order of dims from [E, 1] to [1, E] for concatination along first dim
            y_embeddings = torch.concat((y_embeddings, y_window_filtered[nonzero_filter_indices].reshape(1, E)), dim = 0).to(device)

        x_gt = (x[(k - max_pos_offset - 1):]).to(device)
        # until end

    return y_embeddings, x_gt

# pos offset edition
# added device
# negative filters
def embed_restricted(filter, y, x, max_pos_offset, device):
    """ We are lag-embedding effect Y to later predict cause X. Similarity is defined in the Y manifold, i.e. the embedding space. We also extract corresponding gt X values. This code is designed to work with binary and weighted filters alike. 
    TODO: For loop can probably be avoided to make code faster but embedding is only run once.

    Args:
        filter (k, 0): filter consisting of (a: binary case) 0 and 1 (b: weighted case) weights
        y (N, ): e.g. 401 length timeseries (to be embedded)
        x (N,):  e.g. 401 length timeseries (target)
        max_pos_offset (1,): integer of max_pos_offset

    Returns:
        y_embeddings (M = N - k + 1, E): embedding y into Y x E dimesions by lagging
        x_gt (M = N - k + 1, ): ground truth x values corresponding to each embedding
        E (1): integer of effective embedding dimensions
    """
    # Span / length of the filter including zeros: We expect the number of (complete) embeddings to be d - k + 1 (because if d = k, say 12 = 12, we still have 1 valid sequence)
    k = torch.tensor(filter.shape[0]).to(device)

    if (max_pos_offset + 1) > k:
        print("The offset is too large. It is larger than the embedding span.")

    # Extract effective embedding dim (how many ones/non-zero positions)
    # Make inequality so that code works for negative values which we allow during the optimisation process but we discourage through the design of the loss function
    E = torch.tensor(filter[filter != 0].shape[0]).to(device)

    ### Y EMBEDDINGS ###
    # Intialise empty tensor with the according dim E
    y_embeddings = torch.empty(size = (0, E)).to(device)

    # Plus one because stop in range is excluded
    for i in range(y.shape[0] - k + 1):
        # subset corresponding window (with span k) from y timeseries
        y_window = y[i : (i + k)].to(device)
        # element-wise multiplication
        y_window_filtered = torch.mul(y_window, filter).to(device)
        # extract non-zero elements
        # TODO This step could be weakness if true values are zero. Change implementation if needed
        nonzero_filter_indices = torch.nonzero(filter).to(device)
        # change order of dims from [E, 1] to [1, E] for concatination along first dim
        y_embeddings = torch.concat((y_embeddings, y_window_filtered[nonzero_filter_indices].reshape(1, E)), dim = 0).to(device)

    ### X GROUND TRUTH ###
    # Select the latter x values as ground truth as we have full "history" for these
    # Now goes further back but doesn't reach end
    if max_pos_offset == 0:
        x_gt = x[ - y_embeddings.shape[0]:].to(device)
    else:
        x_gt = (x[ - (y_embeddings.shape[0] + max_pos_offset): - max_pos_offset]).to(device)

    return y_embeddings, x_gt


