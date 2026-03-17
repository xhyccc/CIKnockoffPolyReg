"""Data loader for IC-Knock-Poly comparison experiments.

Supports loading datasets from:
  - CSV files (``pandas`` or ``csv`` module)
  - NumPy ``.npy`` / ``.npz`` files
  - In-memory NumPy arrays

The loader always returns a ``DataBundle`` with named fields so that
all comparison methods receive identical data.

CSV convention
--------------
The last column of the CSV is treated as the response ``y``.  All other
columns are treated as features ``X``.  An optional header row is detected
automatically.

NPZ convention
--------------
The archive must contain at least ``"X"`` and ``"y"`` keys.  An optional
``"X_unlabeled"`` key is loaded when present.

NPY convention
--------------
Two separate files are expected: one for ``X`` (shape ``(n, p)``) and one
for ``y`` (shape ``(n,)``).  Pass both paths to ``DataLoader.from_npy``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class DataBundle:
    """Container for a loaded dataset.

    Attributes
    ----------
    X : ndarray of shape (n_labeled, p)
        Labeled feature matrix.
    y : ndarray of shape (n_labeled,)
        Response vector.
    X_unlabeled : ndarray of shape (N, p) or None
        Additional unlabeled feature observations (semi-supervised mode).
    feature_names : list of str
        Column names for ``X`` (auto-generated if not available in the file).
    source : str
        Path or description of the data source.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X_unlabeled: Optional[NDArray[np.float64]] = None
    feature_names: list = field(default_factory=list)
    source: str = ""

    @property
    def n_labeled(self) -> int:
        """Number of labeled observations."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of base features (columns of X)."""
        return self.X.shape[1]

    @property
    def n_unlabeled(self) -> int:
        """Number of unlabeled observations (0 if none)."""
        return 0 if self.X_unlabeled is None else self.X_unlabeled.shape[0]


class DataLoader:
    """Load datasets for IC-Knock-Poly comparison experiments.

    All class methods are static and return a ``DataBundle``.

    Examples
    --------
    Load from CSV::

        bundle = DataLoader.from_csv("data/experiment.csv")

    Load from NPZ::

        bundle = DataLoader.from_npz("data/experiment.npz")

    Load from two NPY files::

        bundle = DataLoader.from_npy("data/X.npy", "data/y.npy")

    Wrap in-memory arrays::

        bundle = DataLoader.from_arrays(X, y, feature_names=["temp", "pressure"])
    """

    @staticmethod
    def from_csv(
        path: str,
        response_col: Optional[Union[int, str]] = None,
        has_header: bool = True,
        unlabeled_path: Optional[str] = None,
        delimiter: str = ",",
    ) -> DataBundle:
        """Load a labeled dataset from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        response_col : int, str, or None
            Column index (0-based) or name of the response variable.
            Defaults to the **last** column.
        has_header : bool
            Whether the first row is a header.  Default ``True``.
        unlabeled_path : str or None
            Optional path to a second CSV containing only feature columns
            (no response) for semi-supervised mode.
        delimiter : str
            Field delimiter.  Default ``","`` (comma).

        Returns
        -------
        DataBundle
        """
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {path}")

        header: list[str] = []
        if has_header:
            header = rows[0]
            data_rows = rows[1:]
        else:
            data_rows = rows

        if not data_rows:
            raise ValueError(f"CSV file has no data rows: {path}")

        arr = np.array(data_rows, dtype=float)
        n_cols = arr.shape[1]

        # Determine response column index
        if isinstance(response_col, str):
            if not header:
                raise ValueError(
                    "response_col given as string but has_header=False"
                )
            resp_idx = header.index(response_col)
        elif response_col is not None:
            resp_idx = int(response_col)
        else:
            resp_idx = n_cols - 1

        feat_cols = [j for j in range(n_cols) if j != resp_idx]
        X = arr[:, feat_cols]
        y = arr[:, resp_idx]

        feat_names = (
            [header[j] for j in feat_cols] if header else [f"x{j}" for j in feat_cols]
        )

        X_unlabeled = None
        if unlabeled_path is not None:
            with open(unlabeled_path, newline="") as f:
                reader2 = csv.reader(f, delimiter=delimiter)
                u_rows = list(reader2)
            if has_header and u_rows:
                u_rows = u_rows[1:]
            if u_rows:
                X_unlabeled = np.array(u_rows, dtype=float)

        return DataBundle(
            X=X,
            y=y,
            X_unlabeled=X_unlabeled,
            feature_names=feat_names,
            source=path,
        )

    @staticmethod
    def from_npz(path: str) -> DataBundle:
        """Load a dataset from a NumPy ``.npz`` archive.

        The archive must contain keys ``"X"`` (shape ``(n, p)``) and ``"y"``
        (shape ``(n,)``).  Optional keys:

        * ``"X_unlabeled"`` — unlabeled feature matrix for semi-supervised mode.
        * ``"feature_names"`` — 1-D array of strings naming each feature column.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        DataBundle
        """
        archive = np.load(path, allow_pickle=False)
        X = archive["X"].astype(np.float64)
        y = archive["y"].astype(np.float64).ravel()
        X_unlabeled = (
            archive["X_unlabeled"].astype(np.float64)
            if "X_unlabeled" in archive
            else None
        )
        feat_names = (
            list(archive["feature_names"].astype(str))
            if "feature_names" in archive
            else [f"x{j}" for j in range(X.shape[1])]
        )
        return DataBundle(
            X=X, y=y, X_unlabeled=X_unlabeled, feature_names=feat_names, source=path
        )

    @staticmethod
    def from_npy(
        X_path: str,
        y_path: str,
        X_unlabeled_path: Optional[str] = None,
    ) -> DataBundle:
        """Load from separate ``.npy`` files for X and y.

        Parameters
        ----------
        X_path : str
            Path to ``X.npy`` (shape ``(n, p)``).
        y_path : str
            Path to ``y.npy`` (shape ``(n,)``).
        X_unlabeled_path : str or None
            Optional path to an unlabeled feature matrix ``.npy``.

        Returns
        -------
        DataBundle
        """
        X = np.load(X_path).astype(np.float64)
        y = np.load(y_path).astype(np.float64).ravel()
        X_unlabeled = (
            np.load(X_unlabeled_path).astype(np.float64)
            if X_unlabeled_path is not None
            else None
        )
        source = f"{X_path}+{y_path}"
        return DataBundle(
            X=X,
            y=y,
            X_unlabeled=X_unlabeled,
            feature_names=[f"x{j}" for j in range(X.shape[1])],
            source=source,
        )

    @staticmethod
    def from_arrays(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_unlabeled: Optional[NDArray[np.float64]] = None,
        feature_names: Optional[list[str]] = None,
        source: str = "in-memory",
    ) -> DataBundle:
        """Wrap in-memory NumPy arrays in a ``DataBundle``.

        Parameters
        ----------
        X : (n_labeled, p) array
            Labeled feature matrix.
        y : (n_labeled,) array
            Response vector.
        X_unlabeled : (N, p) array or None
            Optional unlabeled feature observations.
        feature_names : list of str or None
            Names for the ``p`` feature columns.
        source : str
            Optional label for the data source.

        Returns
        -------
        DataBundle
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X_unlabeled is not None:
            X_unlabeled = np.asarray(X_unlabeled, dtype=np.float64)
        p = X.shape[1]
        names = feature_names if feature_names is not None else [f"x{j}" for j in range(p)]
        return DataBundle(
            X=X, y=y, X_unlabeled=X_unlabeled, feature_names=names, source=source
        )

    @staticmethod
    def save_npz(
        path: str,
        bundle: DataBundle,
    ) -> None:
        """Save a ``DataBundle`` to a ``.npz`` archive.

        Parameters
        ----------
        path : str
            Destination file path (will be created or overwritten).
        bundle : DataBundle
            Dataset to save.
        """
        arrays: dict[str, np.ndarray] = {
            "X": bundle.X,
            "y": bundle.y,
            "feature_names": np.array(bundle.feature_names, dtype=str),
        }
        if bundle.X_unlabeled is not None:
            arrays["X_unlabeled"] = bundle.X_unlabeled
        np.savez(path, **arrays)
