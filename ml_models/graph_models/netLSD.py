from typing import List, Union

import networkx as nx
import numpy as np

from ml_models.utils import (
    check_1d,
    check_2d,
    eigenvalues_auto,
    graph_to_laplacian,
    mat_to_laplacian,
)


class NetLSD:
    """
    NetLSD embedding model for whole-graph representation learning.

    :param timescales: Vector of discrete timesteps for the kernel computation
    :type timescales: np.ndarray
    :param kernel: Either 'heat' or 'wave'. Type of kernel to use
    :type kernel: str
    :param eigenvalues: Number of eigenvalues to compute / use for approximation
    :type eigenvalues: str, int or tuple
    :param normalization: Either 'empty', 'complete', or None. Can also be a numpy array
    :type normalization: Union[str, np.ndarray, None]
    :param normalized_laplacian: Defines whether the eigenvalues came from normalized Laplacian
    :type normalized_laplacian: bool

    Example
    _______
    from ml_models.graph_models.netLSD import NetLSD
    from dataloader.dataloader import ds_to_graphs

    data = ds_to_graphs("data/MUTAG")
    model = NetLSD(timescales=np.logspace(-2,2,250), kernel='heat')
    embeddings = [model.fit_transform(g) for g in data["graphs"]]
    """

    def __init__(
        self,
        timescales: np.ndarray = np.logspace(-2, 2, 250),
        kernel: str = "heat",
        eigenvalues: Union[str, int, tuple] = "auto",
        normalization: Union[str, np.ndarray, None] = "empty",
        normalized_laplacian: bool = True,
    ) -> None:
        self.timescales = timescales
        self.kernel = kernel
        self.eigenvalues = eigenvalues
        self.normalization = normalization
        self.normalized_laplacian = normalized_laplacian

    def compare(self, descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
        """
        Compute the distance between two NetLSD signatures.

        :param descriptor1: First signature to compare
        :type descriptor1: np.ndarray
        :param descriptor2: Second signature to compare
        :type descriptor2: np.ndarray
        :returns: NetLSD distance
        :rtype: float
        """
        return np.linalg.norm(descriptor1 - descriptor2)

    def fit_transform(self, inp: Union[nx.Graph, np.ndarray]) -> np.ndarray:
        """
        Compute NetLSD signature for a single graph or adjacency matrix.

        :param inp: NetworkX graph or adjacency matrix
        :type inp: Union[nx.Graph, np.ndarray]
        :returns: NetLSD signature
        :rtype: np.ndarray
        """
        return self._netlsd(
            inp,
            self.timescales,
            self.kernel,
            self.eigenvalues,
            self.normalization,
            self.normalized_laplacian,
        )

    def heat(self, inp: Union[nx.Graph, np.ndarray]) -> np.ndarray:
        """
        Compute heat kernel trace signature.

        :param inp: NetworkX graph or adjacency matrix
        :type inp: Union[nx.Graph, np.ndarray]
        :returns: Heat kernel trace signature
        :rtype: np.ndarray
        """
        return self._netlsd(
            inp,
            self.timescales,
            kernel="heat",
            eigenvalues=self.eigenvalues,
            normalization=self.normalization,
            normalized_laplacian=self.normalized_laplacian,
        )

    def wave(self, inp: Union[nx.Graph, np.ndarray]) -> np.ndarray:
        """
        Compute wave kernel trace signature.

        :param inp: NetworkX graph or adjacency matrix
        :type inp: Union[nx.Graph, np.ndarray]
        :returns: Wave kernel trace signature
        :rtype: np.ndarray
        """
        return self._netlsd(
            inp,
            self.timescales,
            kernel="wave",
            eigenvalues=self.eigenvalues,
            normalization=self.normalization,
            normalized_laplacian=self.normalized_laplacian,
        )

    def _netlsd(
        self,
        inp,
        timescales: np.ndarray,
        kernel: str,
        eigenvalues: Union[str, int, tuple],
        normalization: Union[str, np.ndarray, None],
        normalized_laplacian: bool,
    ) -> np.ndarray:
        """
        Compute NetLSD signature from input, timescales, and normalization.

        :param inp: NetworkX graph, adjacency matrix, or eigenvalue vector
        :type inp: object
        :param timescales: Vector of timesteps
        :type timescales: np.ndarray
        :param kernel: Either 'heat' or 'wave'
        :type kernel: str
        :param eigenvalues: Eigenvalue approximation method or count
        :type eigenvalues: str, int or tuple
        :param normalization: Normalization method or vector
        :type normalization: Union[str, np.ndarray, None]
        :param normalized_laplacian: Whether to use normalized Laplacian
        :type normalized_laplacian: bool
        :returns: NetLSD signature
        :rtype: np.ndarray
        """
        if kernel not in {"heat", "wave"}:
            raise AttributeError(f"Unrecognized kernel type: {kernel}")
        if not isinstance(normalized_laplacian, bool):
            raise AttributeError(
                f"Unknown Laplacian type: {type(normalized_laplacian)}"
            )
        if not isinstance(eigenvalues, (int, tuple, str)):
            raise AttributeError(f"Unrecognized eigenvalues type: {type(eigenvalues)}")
        if not isinstance(timescales, np.ndarray) or timescales.ndim != 1:
            raise AttributeError("Timescales must be a 1D numpy array")
        if normalization not in {"complete", "empty", "none", True, False, None}:
            if not isinstance(normalization, np.ndarray) or normalization.ndim != 1:
                raise AttributeError(
                    'Normalization must be "complete", "empty", None, or a 1D numpy array'
                )
            if normalization.shape[0] != timescales.shape[0]:
                raise AttributeError(
                    "Normalization vector length must match timescales length"
                )

        eivals = check_1d(inp)
        if eivals is None:
            mat = check_2d(inp)
            if mat is None:
                mat = graph_to_laplacian(inp, normalized_laplacian)
                if mat is None:
                    raise ValueError(f"Unrecognized input type: {type(inp)}")
            else:
                mat = mat_to_laplacian(inp, normalized_laplacian)
            eivals = eigenvalues_auto(mat, eigenvalues)

        if kernel == "heat":
            return self._hkt(eivals, timescales, normalization, normalized_laplacian)
        else:
            return self._wkt(eivals, timescales, normalization, normalized_laplacian)

    def _hkt(self, eivals, timescales, normalization, normalized_laplacian):
        """
        Compute heat kernel trace from eigenvalues.

        :param eivals: Eigenvalue vector
        :type eivals: np.ndarray
        :param timescales: Vector of timesteps
        :type timescales: np.ndarray
        :param normalization: Normalization method or vector
        :type normalization: Union[str, np.ndarray, None]
        :param normalized_laplacian: Whether to use normalized Laplacian
        :type normalized_laplacian: bool
        :returns: Heat kernel trace signature
        :rtype: np.ndarray
        """
        nv = eivals.shape[0]
        hkt = np.zeros(timescales.shape)
        for idx, t in enumerate(timescales):
            hkt[idx] = np.sum(np.exp(-t * eivals))
        if isinstance(normalization, np.ndarray):
            return hkt / normalization
        if normalization == "empty" or normalization is True:
            return hkt / nv
        if normalization == "complete":
            if normalized_laplacian:
                return hkt / (1 + (nv - 1) * np.exp(-timescales))
            else:
                return hkt / (1 + nv * np.exp(-nv * timescales))
        return hkt

    def _wkt(self, eivals, timescales, normalization, normalized_laplacian):
        """
        Compute wave kernel trace from eigenvalues.

        :param eivals: Eigenvalue vector
        :type eivals: np.ndarray
        :param timescales: Vector of timesteps
        :type timescales: np.ndarray
        :param normalization: Normalization method or vector
        :type normalization: Union[str, np.ndarray, None]
        :param normalized_laplacian: Whether to use normalized Laplacian
        :type normalized_laplacian: bool
        :returns: Wave kernel trace signature
        :rtype: np.ndarray
        """
        nv = eivals.shape[0]
        wkt = np.zeros(timescales.shape, dtype=complex)
        for idx, t in enumerate(timescales):
            wkt[idx] = np.sum(np.exp(-1j * t * eivals))
        if isinstance(normalization, np.ndarray):
            return wkt / normalization
        if normalization == "empty" or normalization is True:
            return wkt / nv
        if normalization == "complete":
            if normalized_laplacian:
                return wkt / (1 + (nv - 1) * np.cos(timescales))
            else:
                return wkt / (1 + (nv - 1) * np.cos(nv * timescales))
        return wkt
