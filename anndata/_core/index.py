import collections.abc as cabc
from functools import singledispatch
from itertools import repeat
from typing import Union, Sequence, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse


Index1D = Union[slice, int, str, np.int64, np.ndarray]
Index = Union[Index1D, Tuple[Index1D, Index1D], spmatrix]


def get_vector(adata, k, coldim, idxdim, layer=None):
    # adata could be self if Raw and AnnData shared a parent
    dims = ("obs", "var")
    col = getattr(adata, coldim).columns
    idx = getattr(adata, f"{idxdim}_names")

    in_col = k in col
    in_idx = k in idx

    if (in_col + in_idx) == 2:
        raise ValueError(
            f"Key {k} could be found in both .{idxdim}_names and .{coldim}.columns"
        )
    elif (in_col + in_idx) == 0:
        raise KeyError(
            f"Could not find key {k} in .{idxdim}_names or .{coldim}.columns."
        )
    elif in_col:
        return getattr(adata, coldim)[k].values
    elif in_idx:
        selected_dim = dims.index(idxdim)
        idx = adata._normalize_indices(make_slice(k, selected_dim))
        a = adata._get_X(layer=layer)[idx]
    if issparse(a):
        a = a.toarray()
    return np.ravel(a)
