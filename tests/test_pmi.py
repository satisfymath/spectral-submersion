"""Tests for PMI module."""

import numpy as np

from spectral_submersion.pmi import ppmi_matrix


def test_ppmi_positive():
    C = np.array([[2, 1], [1, 2]], dtype=float)
    P = ppmi_matrix(C)
    assert np.all(P >= 0)


def test_ppmi_diagonal_high():
    # Diagonal dominance should yield high PPMI on diagonal
    C = np.eye(3) * 10 + 1
    P = ppmi_matrix(C)
    diag = np.diag(P)
    off_diag = P[~np.eye(3, dtype=bool)]
    assert np.all(diag > off_diag.max())
