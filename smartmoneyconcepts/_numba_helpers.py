import numpy as np
from numba import njit


@njit(cache=True)
def _fvg_causal(high, low, close, open_, n, join_consecutive):
    """
    Combined FVG detection + join_consecutive + bar-by-bar mitigation.
    All three stages in one njit call to avoid Python↔njit overhead.

    Returns: (fvg, top, bottom, mitigated_index)
    """
    fvg = np.full(n, np.nan)
    top = np.full(n, np.nan)
    bottom = np.full(n, np.nan)

    # --- Stage 1: Detection (3-bar pattern) ---
    for j in range(2, n):
        if high[j - 2] < low[j] and close[j - 1] > open_[j - 1]:
            fvg[j] = 1.0
            top[j] = low[j]
            bottom[j] = high[j - 2]
        elif low[j - 2] > high[j] and close[j - 1] < open_[j - 1]:
            fvg[j] = -1.0
            top[j] = low[j - 2]
            bottom[j] = high[j]

    # --- Stage 2: Join consecutive ---
    if join_consecutive:
        for i in range(n - 1):
            if fvg[i] == fvg[i + 1]:
                if top[i] > top[i + 1]:
                    top[i + 1] = top[i]
                if bottom[i] < bottom[i + 1]:
                    bottom[i + 1] = bottom[i]
                fvg[i] = np.nan
                top[i] = np.nan
                bottom[i] = np.nan

    # --- Stage 3: Bar-by-bar mitigation ---
    mitigated_index = np.zeros(n, dtype=np.int32)
    active = np.empty(n, dtype=np.int64)
    n_active = 0
    fvg_dir = np.zeros(n, dtype=np.int32)

    for j in range(n):
        # Check all active FVGs against current bar
        for k in range(n_active):
            idx = active[k]
            if fvg_dir[idx] == 1 and low[j] <= top[idx]:
                mitigated_index[idx] = np.int32(j)
                active[k] = -1
            elif fvg_dir[idx] == -1 and high[j] >= bottom[idx]:
                mitigated_index[idx] = np.int32(j)
                active[k] = -1

        # Compact active array (remove -1 entries)
        write = 0
        for k in range(n_active):
            if active[k] != -1:
                active[write] = active[k]
                write += 1
        n_active = write

        # Add new FVG to active list
        if not np.isnan(fvg[j]):
            fvg_dir[j] = np.int32(fvg[j])
            active[n_active] = j
            n_active += 1

    return fvg, top, bottom, mitigated_index
