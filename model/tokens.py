"""

This file is created under the CENG 502 Project.


"""

import torch
from scipy.spatial.distance import cdist

# num_tokens is K in the algorithm 1 (not the K of K-shot images)
# Please refer to Algorithm 1 given in Supplementary Material
# to match the notation of variables in the comments.
# We assume X, locations of foreground pixels, is the locations of
# f_s, i..e foreground support pixel features.


# Shapes:
# X --> [batchsize, num_foreground_pixels, 2] --> edit: since number of foreground pixels change for every image, X is a list with len(X)=batchsize
# L --> [batchsize, num_background_pixels, 2] last dimension is (x,y) location
# f_s --> [batchsize, h, w, c] 
# "h, w denote the height, width of the feature map." (Supplementary Material)
def init_agent_tokens(num_tokens, X, L, f_s):
  
    # Compute euclidean distance between every pair
    # (foreground_pixel, bacground_pixel)
    # in total, |X| x |L| pairs
    #dists_batch = torch.cdist(X, L)   # Get all the distances for K support ims

    tokens = torch.empty((len(X), num_tokens, f_s.shape[1]))
    L_new = []
    # TODO: can we compute this jointly for all images in a batch?
    for i in range(len(X)):
        L_single = L[i]      # L for a single image in a batch

        for k in range(num_tokens):
            #dists = dists_batch[i]
            dists = torch.from_numpy(cdist(X[i], L[i], 'euclidean'))   # Get all the distances for K support ims

            # See line 3 of Algorithm 1 in Supplementary Material:
            # for a specific location x, min distance between x and all other locations in L
            d_x, d_x_ind = torch.min(dists, dim=1)  

            # We don't care about the actual distance value, so it is named as _
            # we care about which location has the furthest distance p* 
            _ , p_ind = torch.max(d_x, dim=0)

            p_furthest = X[i][p_ind, :]      # This is a location (x,y) of a pixel
            p_star = p_furthest.unsqueeze(0) # [2] --> [1,2] 
            L_single = torch.cat([L_single, p_star], dim=0) # L = (B) U (P), see line 5 in Algorithm 1

            f_a_k = f_s[i, :, p_furthest.data[0].long().item(), p_furthest.data[1].long().item()]
            
            tokens[i,k,:] = f_a_k
            
        L_new.append(L_single)
    
    return tokens
    