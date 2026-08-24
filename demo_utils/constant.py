
import torch
height = 480
width = 832

def GET_ZERO_VAE_CACHE(height, width):
    """
    Build the list of zero tensors for the VAE cache at the given resolution.
    
    Args:
        height: image height
        width: image width
    
    Returns:
        a list of zero tensors, one per resolution level
    """
    return [
        torch.zeros(1, 16, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 384, 2, height//8, width//8),
        torch.zeros(1, 192, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 384, 2, height//4, width//4),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 192, 2, height//2, width//2),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width),
        torch.zeros(1, 96, 2, height, width)
    ]

ZERO_VAE_CACHE = GET_ZERO_VAE_CACHE(height, width)

feat_names = [f"vae_cache_{i}" for i in range(len(ZERO_VAE_CACHE))]
ALL_INPUTS_NAMES = ["z", "use_cache"] + feat_names
