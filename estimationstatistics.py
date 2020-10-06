from typing import Sequence
import numpy as np


def mahalanobis_distance_squared(
    # shape (n,)
    a: np.ndarray,
    # shape (n,)
    b: np.ndarray,
    # shape (n, n)
    psd_mat: np.ndarray,
) -> float:  # positive
    diff = a - b
    dist = diff @ np.linalg.solve(psd_mat, diff)
    return dist


def NEES_sequence(
    # shape (N, n)
    mean_seq,
    # shape (N, n, n)
    cov_seq,
    # shape (N, n)
    true_seq,
) -> np.ndarray:
    """ Batch calculate NEES """
    NEES_seq = np.array(
        [
            mahalanobis_distance_squared(mean, true, cov)
            for mean, true, cov in zip(mean_seq, true_seq, cov_seq)
        ]
    )
    return NEES_seq


def NEES_sequence_indexed(
    # shape (N, n)
    mean_seq,
    # shape (N, n, n)
    cov_seq,
    # shape (N, n)
    true_seq,
    # into which part of the state to calculate NEES for
    idxs: Sequence[int],
) -> np.ndarray:
    mean_seq_indexed = mean_seq[:, idxs]
    # raise NotImplementedError  # np.c_ seems to not be the right thing here!
    cov_seq_indexed = cov_seq[:, idxs][:, :, idxs]
    true_seq_indexed = true_seq[:, idxs]

    NEES_seq = NEES_sequence(mean_seq_indexed, cov_seq_indexed, true_seq_indexed)
    return NEES_seq


def distance_sequence(
    # shape (N, n)
    mean_seq,
    # shape (N, n)
    true_seq,
) -> np.array:
    dists = np.linalg.norm(mean_seq - true_seq, axis=1)
    return dists


def distance_sequence_indexed(
    # shape (N, n)
    mean_seq,
    # shape (N, n)
    true_seq,
    # into which part of the state to calculate NEES for
    idxs: Sequence[int],
) -> np.ndarray:
    mean_seq_indexed = mean_seq[:, idxs]
    true_seq_indexed = true_seq[:, idxs]
    dists = distance_sequence(mean_seq_indexed, true_seq_indexed)
    return dists
