import torch
import torch.nn.functional as F
import numpy as np

def compare_dist(source, target, warp_funct):
    """
    To compare distribution between to foreground data using KL-divergece. This function will do foreground warping to image first before computing the KL-div
    Args:
        source: Source data to be compared. Numpy data with shape = [N, 5], with N denoting number of data points and 5 for xyrgb data
        target: Target data to be compared. Numpy data with shape similar to source
        warp_funct: Warping function
    Result:
        kldiv: The KL-divergene result. Scalar value. 
    """
    # warp the foreground data into the pre-defined image size
    source = warp_funct(source) #[H, W, 3]
    target = warp_funct(target) #[H, W, 3]

    # add small value for all-zeros to avoid undefined log values
    source = source.reshape((-1,3)).astype(np.float) #[n_pixels, 3]
    source[source==0] = source[source==0] + 1e-10
    target = target.reshape((-1,3)).astype(np.float) #[n_pixels, 3]
    target[target==0] = target[target==0] + 1e-10

    # compute the kl-div
    kldiv = F.kl_div(input=F.log_softmax(torch.from_numpy(source), dim=-1),
                     target=F.softmax(torch.from_numpy(target), dim=-1),
                     reduction='batchmean') # scalar
    return np.around(kldiv, decimals=4)