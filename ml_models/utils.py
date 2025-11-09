from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import torch.nn as nn


def init_weights(net: nn.Module) -> None:
    """
    Initialize a torch model with kaiming weights

    :param net: the input model
    :type net: nn.Module
    """
    for block in net:
        for layer in block.children():
            for name, weight in layer.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(weight)
                if "bias" in name:
                    nn.init.constant_(weight, 0.0)


def check_1d(inp: object) -> Optional[np.ndarray]:
    """
    Check that input is a 1D vector. Converts lists to np.ndarray.

    :param inp: Input vector
    :type inp: object
    :returns: Input as a 1D numpy array or None if not a vector
    :rtype: np.ndarray or None
    """
    if isinstance(inp, list):
        return check_1d(np.array(inp))
    if isinstance(inp, np.ndarray):
        if inp.ndim == 1:
            return inp
    return None


def check_2d(inp: object) -> Optional[Union[np.ndarray, sps.spmatrix]]:
    """
    Check that input is a 2D matrix. Converts lists of lists to np.ndarray.

    :param inp: Input matrix
    :type inp: object
    :returns: Input as a 2D numpy array, sparse matrix, or None
    :rtype: numpy.ndarray, scipy.sparse matrix, or None
    """
    if isinstance(inp, list):
        return check_2d(np.array(inp))
    if isinstance(inp, (np.ndarray, np.matrixlib.defmatrix.matrix)):
        if inp.ndim == 2:
            return inp
    if sps.issparse(inp):
        if inp.ndim == 2:
            return inp
    return None


def graph_to_laplacian(
    G: object, normalized: bool = True
) -> Optional[Union[np.ndarray, sps.spmatrix]]:
    """
    Convert a graph from popular Python packages to a Laplacian matrix.

    Supports NetworkX, graph_tool, and igraph graphs.

    :param G: Input graph
    :type G: object
    :param normalized: Whether to use normalized Laplacian
    :type normalized: bool
    :returns: Laplacian matrix of the input graph
    :rtype: scipy.sparse or np.ndarray, or None if unsupported
    """
    try:
        import networkx as nx

        if isinstance(G, nx.Graph):
            if normalized:
                return nx.normalized_laplacian_matrix(G)
            else:
                return nx.laplacian_matrix(G)
    except ImportError:
        pass
    try:
        import graph_tool.all as gt

        if isinstance(G, gt.Graph):
            if normalized:
                return gt.laplacian_type(G, normalized=True)
            else:
                return gt.laplacian(G)
    except ImportError:
        pass
    try:
        import igraph as ig

        if isinstance(G, ig.Graph):
            if normalized:
                return np.array(G.laplacian(normalized=True))
            else:
                return np.array(G.laplacian())
    except ImportError:
        pass
    return None


def mat_to_laplacian(
    mat: Union[np.ndarray, sps.spmatrix], normalized: bool
) -> Union[np.ndarray, sps.spmatrix]:
    """
    Convert an adjacency matrix to a Laplacian matrix.

    If input is already a Laplacian, it is returned unchanged.

    :param mat: Input adjacency matrix
    :type mat: numpy.ndarray or scipy.sparse
    :param normalized: Whether to use normalized Laplacian
    :type normalized: bool
    :returns: Laplacian of the input adjacency matrix
    :rtype: numpy.ndarray or scipy.sparse
    """
    if sps.issparse(mat):
        if np.all(mat.diagonal() >= 0):
            if np.all((mat - sps.diags(mat.diagonal())).data <= 0):
                return mat
    else:
        if np.all(np.diag(mat) >= 0):
            if np.all(mat - np.diag(mat) <= 0):
                return mat
    deg = np.squeeze(np.asarray(mat.sum(axis=1)))
    if sps.issparse(mat):
        L = sps.diags(deg) - mat
    else:
        L = np.diag(deg) - mat
    if not normalized:
        return L
    with np.errstate(divide="ignore"):
        sqrt_deg = 1.0 / np.sqrt(deg)
    sqrt_deg[sqrt_deg == np.inf] = 0
    if sps.issparse(mat):
        sqrt_deg_mat = sps.diags(sqrt_deg)
    else:
        sqrt_deg_mat = np.diag(sqrt_deg)
    return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


def updown_linear_approx(
    eigvals_lower: np.ndarray, eigvals_upper: np.ndarray, nv: int
) -> np.ndarray:
    """
    Approximate Laplacian spectrum using lower and upper parts of eigenvalues.

    :param eigvals_lower: Lower part of the spectrum, sorted
    :type eigvals_lower: numpy.ndarray
    :param eigvals_upper: Upper part of the spectrum, sorted
    :type eigvals_upper: numpy.ndarray
    :param nv: Total number of nodes in the graph
    :type nv: int
    :returns: Approximated eigenvalue vector
    :rtype: numpy.ndarray
    """
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError(
            f"Number of supplied eigenvalues ({nal} lower and {nau} upper) is higher than number of nodes ({nv})!"
        )
    ret = np.zeros(nv)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    ret[nal - 1 : -nau + 1] = np.linspace(
        eigvals_lower[-1], eigvals_upper[0], nv - nal - nau + 2
    )
    return ret


def eigenvalues_auto(
    mat: Union[np.ndarray, sps.spmatrix],
    n_eivals: Union[str, int, Tuple[int, int]] = "auto",
) -> np.ndarray:
    """
    Automatically compute eigenvalues of a Laplacian matrix with approximation.

    :param mat: Laplacian matrix
    :type mat: numpy.ndarray or scipy.sparse
    :param n_eivals: Number of eigenvalues to compute or approximation method
    :type n_eivals: str, int, or tuple
    :returns: Eigenvalue vector (approximated if needed)
    :rtype: numpy.ndarray
    """
    do_full = True
    n_lower = 150
    n_upper = 150
    nv = mat.shape[0]
    if n_eivals == "auto":
        if mat.shape[0] > 1024:
            do_full = False
    if n_eivals == "full":
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        do_full = False
    if do_full and sps.issparse(mat):
        mat = mat.todense()
    if sps.issparse(mat):
        if n_lower == n_upper:
            tr_eivals = spsl.eigsh(
                mat, 2 * n_lower, which="BE", return_eigenvectors=False
            )
            return updown_linear_approx(tr_eivals[:n_upper], tr_eivals[n_upper:], nv)
        else:
            lo_eivals = spsl.eigsh(mat, n_lower, which="SM", return_eigenvectors=False)[
                ::-1
            ]
            up_eivals = spsl.eigsh(mat, n_upper, which="LM", return_eigenvectors=False)
            return updown_linear_approx(lo_eivals, up_eivals, nv)
    else:
        if do_full:
            return spl.eigvalsh(mat)
        else:
            lo_eivals = spl.eigvalsh(mat, eigvals=(0, n_lower - 1))
            up_eivals = spl.eigvalsh(mat, eigvals=(nv - n_upper - 1, nv - 1))
            return updown_linear_approx(lo_eivals, up_eivals, nv)
