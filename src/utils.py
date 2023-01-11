import numpy as np
import torch


def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))


def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

# Thanks to https://github.com/christiansiegel/coding-theory-algorithms/blob/master/LinearBlockCode.py


def GtoP(G):
    """Extract the submatrix P from a Generator Matrix in systematic form.
    Args:
        G: Generator Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    k = G.shape[0]
    n = G.shape[1]
    P = G[:k, :n - k]
    return P


def HtoSystematicH(H):
    """Convert a parity check matrix into systematic form,
    gotten from https://gist.github.com/popcornell/bc29d1b7ba37d824335ab7b6280f7fec"""
    m, n = H.shape

    i = 0
    j = 0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(H[i:, j]) + i

        # swap rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(H[k])
        H[k] = H[i]
        H[i] = temp

        aijn = H[i, j:]

        # make a copy otherwise M will be directly affected
        col = np.copy(H[:, j])

        col[i] = 0  # avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        H[:, j:] = H[:, j:] ^ flip

        i += 1
        j += 1

    return H


def HtoP(H):
    """Extract the submatrix P from a Parity Check Matrix in systematic form.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    n = H.shape[1]
    k = n - H.shape[0]
    PK = H[:, n - k:n]
    P = PK.transpose(0, 1)
    return P


def HtoG(H):
    """Convert a Parity Check Matrix in systematic form to a Generator Matrix.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Generator Matrix G
    """
    n = H.shape[1]
    k = n - H.shape[0]
    P = HtoP(H)
    Ik = np.eye(k)
    G = np.concatenate((P, Ik), axis=0)
    return G
