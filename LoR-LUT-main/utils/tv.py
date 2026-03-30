
import torch
import torch.nn.functional as F

def tv_3d(L):
    """
    Total variation over a 3D LUT volume.
    L: [B,G,G,G,3]
    """
    dx = L[:,1:,:,:,:] - L[:,:-1,:,:,:]
    dy = L[:,:,1:,:,:] - L[:,:, :-1,:,:]
    dz = L[:,:,:,1:,:] - L[:,:,:, :-1,:]
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())

def l2_residual(delta):
    return (delta**2).mean()

def monotonicity_3d(L):
    """
    Monotonicity regularization over a 3D LUT volume.
    Penalizes negative slopes along each input axis for all output channels.
    L: [B,G,G,G,3]
    return: scalar penalty (mean of negative parts)
    """
    dx = L[:, 1:, :, :, :] - L[:, :-1, :, :, :]
    dy = L[:, :, 1:, :, :] - L[:, :, :-1, :, :]
    dz = L[:, :, :, 1:, :] - L[:, :, :, :-1, :]
    # Negative part penalty (ReLU on the negated diffs)
    px = F.relu(-dx).mean()
    py = F.relu(-dy).mean()
    pz = F.relu(-dz).mean()
    return px + py + pz
