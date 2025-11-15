from typing import Tuple, Union

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
        eigenvalues: Union[str, int, Tuple[int, int]] = "auto",
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
        return float(np.linalg.norm(descriptor1 - descriptor2))

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
        inp: Union[nx.Graph, np.ndarray],
        timescales: np.ndarray,
        kernel: str,
        eigenvalues: Union[str, int, Tuple[int, int]],
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
            raise AttributeError(
                "Unirecognized kernel type: expected one of ['heat', 'wave'], got {0}".format(
                    kernel
                )
            )
        if not isinstance(normalized_laplacian, bool):
            raise AttributeError(
                "Unknown Laplacian type: expected bool, got {0}".format(
                    normalized_laplacian
                )
            )
        if not isinstance(eigenvalues, (int, tuple, str)):
            raise AttributeError(
                "Unirecognized requested eigenvalue number: expected type of ['str', 'tuple', or 'int'], got {0}".format(
                    type(eigenvalues)
                )
            )
        if not isinstance(timescales, np.ndarray):
            raise AttributeError(
                "Unirecognized timescales data type: expected np.ndarray, got {0}".format(
                    type(timescales)
                )
            )
        if timescales.ndim != 1:
            raise AttributeError(
                "Unirecognized timescales dimensionality: expected a vector, got {0}-d array".format(
                    timescales.ndim
                )
            )
        if normalization not in {"complete", "empty", "none", True, False, None}:
            if not isinstance(normalization, np.ndarray):
                raise AttributeError(
                    "Unirecognized normalization type: expected one of ['complete', 'empty', None or np.ndarray], got {0}".format(
                        normalization
                    )
                )
            if normalization.ndim != 1:
                raise AttributeError(
                    "Unirecognized normalization dimensionality: expected a vector, got {0}-d array".format(
                        normalization.ndim
                    )
                )
            if timescales.shape[0] != normalization.shape[0]:
                raise AttributeError(
                    "Unirecognized normalization dimensionality: expected {0}-length vector, got length {1}".format(
                        timescales.shape[0], normalization.shape[0]
                    )
                )

        eivals = check_1d(inp)
        if eivals is None:
            mat = check_2d(inp)
            if mat is None:
                mat = graph_to_laplacian(inp, normalized_laplacian)
                if mat is None:
                    raise ValueError(
                        "Unirecognized input type: expected one of ['np.ndarray', 'scipy.sparse', 'networkx.Graph',' graph_tool.Graph,' or 'igraph.Graph'], got {0}".format(
                            type(inp)
                        )
                    )
            else:
                mat = mat_to_laplacian(inp, normalized_laplacian)
            eivals = eigenvalues_auto(mat, eigenvalues)
        if kernel == "heat":
            return self._hkt(eivals, timescales, normalization, normalized_laplacian)
        else:
            return self._wkt(eivals, timescales, normalization, normalized_laplacian)

    def _hkt(
        self,
        eivals: np.ndarray,
        timescales: np.ndarray,
        normalization: Union[str, np.ndarray, None],
        normalized_laplacian: bool,
    ) -> np.ndarray:
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

    def _wkt(
        self,
        eivals: np.ndarray,
        timescales: np.ndarray,
        normalization: Union[str, np.ndarray, None],
        normalized_laplacian: bool,
    ) -> np.ndarray:
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
