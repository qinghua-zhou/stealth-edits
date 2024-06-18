import os

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def pca_torch(
        matrices, 
        n_components='ratio', 
        alphaRatio=0.1, 
        whiten=True, 
        centre=False,
        return_all=True, 
        out=False, 
        device=None,
        eps=1e-12,
        empty_cache=False,
        to_numpy=False,
        return_org=False
    ):
    """ PCA function with torch (with whitening option)
    """
    if type(matrices) != list:
        matrices = [matrices]

    if device is not None:
        matrices = [matrices[i].to(device) for i in range(len(matrices))]

    if centre:
        centroid = torch.mean(matrices[0], dim=0)
        matrices = [matrices[i] - centroid for i in range(len(matrices))]

    # convariance matrix
    cov = torch.cov(matrices[0].T)

    # eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # sort by eigenvalues
    eigenvalues, perm = torch.sort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, perm]
    eigenvalues[eigenvalues < 0] = 0  # Remove any (incorrect) negative eigenvalues

    # select components
    if n_components=='Kaiser':
        n_components = torch.sum(eigenvalues>=1).cpu().item()
        if out: print('Kaiser rule: n_components =', n_components)
    elif n_components=='ratio':
        threshold = alphaRatio * eigenvalues.max()
        n_components = torch.sum(eigenvalues>threshold).cpu().item()
    if out: print('Kaiser rule (ratio 0.1): n_components =', n_components)
    else:
        if out: print('Specified n_components =', n_components)


    if return_org:
        return eigenvectors, eigenvalues
        
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:,:n_components]

    # whiten
    if whiten:
        eigenvectors = eigenvectors * 1.0 / torch.sqrt(eigenvalues + eps)

    # transform
    transformed = [torch.matmul(matrices[i], eigenvectors) for i in range(len(matrices))]

    if to_numpy:
        transformed = [t.detach().cpu().numpy() for t in transformed]
        eigenvectors = eigenvectors.detach().cpu().numpy()
        eigenvalues = eigenvalues.detach().cpu().numpy()

    if return_all:
        return transformed[0], eigenvectors, eigenvalues, transformed
 
    if empty_cache:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        cov = cov.detach()
        del matrices, cov, eigenvalues, eigenvectors, perm

    return transformed


def calc_sep_intrinsic_dim(features, centre=True, eps=0, deltas=0, device=None, return_p=False):
    """
    Compute the separability-based intrinsic dimension of a dataset.
    
    Based on Oliver Sutton's source code for "Relative intrinsic dimensionality is intrinsic to learning" (2023)
    """
    import torch 
    
    if centre:
        centre = torch.mean(features, dim=0)
        features = features - centre

    if device is not None:
        features = features.to(device)

    n_points = features.shape[0]

    # Compute the pairwise projections (x – y, y) as (x, y) - |y|^2
    projections = features @ features.T
    norms_sq = torch.diag(projections)
    projections -= norms_sq

    # Compute the probability of separating pairs of points (count_zero faster for CPU, sum faster for GPU)
    if type(deltas) == int: 
        p = (torch.sum(projections >= deltas) - n_points) / (n_points * (n_points - 1))

        # Convert the probability into a dimensionality
        dim = -1 - torch.log2(p + eps) # find min of p instead

        p = p.detach().cpu().item()
        dim = dim.detach().cpu().item()

    else:
        p = torch.stack([
            (torch.sum(projections >= delta) - n_points) / (n_points * (n_points - 1))
            for delta in deltas
        ])

        # Convert the probability into a dimensionality
        dim = -1 - torch.log2(p + eps) # find min of p instead

        p = p.detach().cpu().numpy()
        dim = dim.detach().cpu().numpy()

    if return_p: 
        return dim, p
    return dim


