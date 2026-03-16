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


@njit(cache=True)
def _bos_choch_causal_break(bos, choch, level, break_high_arr, break_low_arr, n):
    """
    Bar-by-bar break validation + supersession for causal BOS/CHOCH.

    MODIFIES bos, choch, level IN PLACE (supersession zeroes them).
    Returns: broken array
    """
    # Collect active pattern indices
    pattern_buf = np.empty(n, dtype=np.int64)
    n_patterns = 0
    for i in range(n):
        if bos[i] != 0 or choch[i] != 0:
            pattern_buf[n_patterns] = i
            n_patterns += 1

    broken = np.zeros(n, dtype=np.int32)

    for bar in range(n):
        for p in range(n_patterns):
            i = pattern_buf[p]
            if i == -1:
                continue
            if bar <= i + 1:
                continue

            matched = False
            if (bos[i] == 1 or choch[i] == 1) and break_high_arr[bar] > level[i]:
                matched = True
            elif (bos[i] == -1 or choch[i] == -1) and break_low_arr[bar] < level[i]:
                matched = True

            if matched:
                broken[i] = np.int32(bar)
                # Supersession: zero earlier patterns broken at same or later bar
                for q in range(n_patterns):
                    k = pattern_buf[q]
                    if k == -1:
                        continue
                    if k < i and broken[k] >= bar:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0.0
                pattern_buf[p] = -1

        # Compact pattern buffer
        write = 0
        for p in range(n_patterns):
            if pattern_buf[p] != -1:
                pattern_buf[write] = pattern_buf[p]
                write += 1
        n_patterns = write

    return broken


@njit(cache=True)
def _ob_causal_loop(
    ohlc_len, _open, _high, _low, _close, _volume,
    swing_high_indices, swing_low_indices,
    causal_half, close_mitigation, causal,
):
    """
    Full OB detection: bullish loop then bearish loop.

    Returns: (ob, top_arr, bottom_arr, obVolume, lowVolume, highVolume,
              percentage, mitigated_index, breaker)
    """
    crossed = np.full(ohlc_len, False)
    ob = np.zeros(ohlc_len, dtype=np.int32)
    top_arr = np.zeros(ohlc_len, dtype=np.float64)
    bottom_arr = np.zeros(ohlc_len, dtype=np.float64)
    obVolume = np.zeros(ohlc_len, dtype=np.float64)
    lowVolume = np.zeros(ohlc_len, dtype=np.float64)
    highVolume = np.zeros(ohlc_len, dtype=np.float64)
    percentage = np.zeros(ohlc_len, dtype=np.float64)
    mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
    breaker = np.full(ohlc_len, False)

    # ========== Bullish OB loop ==========
    active = np.empty(ohlc_len, dtype=np.int64)
    n_active = 0

    for i in range(ohlc_len):
        # -- Check active bullish OBs --
        for k in range(n_active):
            idx = active[k]
            if breaker[idx]:
                if not causal:
                    if _high[i] > top_arr[idx]:
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active[k] = -1
                else:
                    active[k] = -1
            else:
                mitigated = False
                if not close_mitigation:
                    if _low[i] < bottom_arr[idx]:
                        mitigated = True
                else:
                    body_low = min(_open[i], _close[i])
                    if body_low < bottom_arr[idx]:
                        mitigated = True
                if mitigated:
                    breaker[idx] = True
                    mitigated_index[idx] = np.int32(i - 1)

        # Compact
        write = 0
        for k in range(n_active):
            if active[k] != -1:
                active[write] = active[k]
                write += 1
        n_active = write

        # -- Find last confirmed swing high --
        if causal_half > 0:
            limit_idx = np.searchsorted(swing_high_indices, i - causal_half + 1)
        else:
            limit_idx = np.searchsorted(swing_high_indices, i)

        last_top_index = -1
        if limit_idx > 0:
            last_top_index = swing_high_indices[limit_idx - 1]

        if last_top_index >= 0:
            if _close[i] > _high[last_top_index] and not crossed[last_top_index]:
                crossed[last_top_index] = True
                default_index = i - 1
                obBtm = _high[default_index]
                obTop = _low[default_index]
                obIndex = default_index

                if i - last_top_index > 1:
                    start = last_top_index + 1
                    end = i
                    if end > start:
                        # Last occurrence of minimum low in segment
                        min_val = _low[start]
                        min_idx = start
                        for s in range(start + 1, end):
                            if _low[s] <= min_val:
                                min_val = _low[s]
                                min_idx = s
                        obBtm = _low[min_idx]
                        obTop = _high[min_idx]
                        obIndex = min_idx

                store_idx = i if causal else obIndex
                ob[store_idx] = 1
                top_arr[store_idx] = obTop
                bottom_arr[store_idx] = obBtm
                vol_cur = _volume[i]
                vol_prev1 = _volume[i - 1] if i >= 1 else 0.0
                vol_prev2 = _volume[i - 2] if i >= 2 else 0.0
                obVolume[store_idx] = vol_cur + vol_prev1 + vol_prev2
                lowVolume[store_idx] = vol_prev2
                highVolume[store_idx] = vol_cur + vol_prev1
                max_vol = max(highVolume[store_idx], lowVolume[store_idx])
                if max_vol != 0.0:
                    percentage[store_idx] = (
                        min(highVolume[store_idx], lowVolume[store_idx])
                        / max_vol
                        * 100.0
                    )
                else:
                    percentage[store_idx] = 100.0
                active[n_active] = store_idx
                n_active += 1

    # ========== Bearish OB loop ==========
    n_active = 0  # reset for bearish

    for i in range(ohlc_len):
        # -- Check active bearish OBs --
        for k in range(n_active):
            idx = active[k]
            if breaker[idx]:
                if not causal:
                    if _low[i] < bottom_arr[idx]:
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active[k] = -1
                else:
                    active[k] = -1
            else:
                mitigated = False
                if not close_mitigation:
                    if _high[i] > top_arr[idx]:
                        mitigated = True
                else:
                    body_high = max(_open[i], _close[i])
                    if body_high > top_arr[idx]:
                        mitigated = True
                if mitigated:
                    breaker[idx] = True
                    mitigated_index[idx] = np.int32(i)

        # Compact
        write = 0
        for k in range(n_active):
            if active[k] != -1:
                active[write] = active[k]
                write += 1
        n_active = write

        # -- Find last confirmed swing low --
        if causal_half > 0:
            limit_idx = np.searchsorted(swing_low_indices, i - causal_half + 1)
        else:
            limit_idx = np.searchsorted(swing_low_indices, i)

        last_btm_index = -1
        if limit_idx > 0:
            last_btm_index = swing_low_indices[limit_idx - 1]

        if last_btm_index >= 0:
            if _close[i] < _low[last_btm_index] and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                default_index = i - 1
                obTop = _high[default_index]
                obBtm = _low[default_index]
                obIndex = default_index

                if i - last_btm_index > 1:
                    start = last_btm_index + 1
                    end = i
                    if end > start:
                        # Last occurrence of maximum high in segment
                        max_val = _high[start]
                        max_idx = start
                        for s in range(start + 1, end):
                            if _high[s] >= max_val:
                                max_val = _high[s]
                                max_idx = s
                        obTop = _high[max_idx]
                        obBtm = _low[max_idx]
                        obIndex = max_idx

                store_idx = i if causal else obIndex
                ob[store_idx] = -1
                top_arr[store_idx] = obTop
                bottom_arr[store_idx] = obBtm
                vol_cur = _volume[i]
                vol_prev1 = _volume[i - 1] if i >= 1 else 0.0
                vol_prev2 = _volume[i - 2] if i >= 2 else 0.0
                obVolume[store_idx] = vol_cur + vol_prev1 + vol_prev2
                lowVolume[store_idx] = vol_cur + vol_prev1
                highVolume[store_idx] = vol_prev2
                max_vol = max(highVolume[store_idx], lowVolume[store_idx])
                if max_vol != 0.0:
                    percentage[store_idx] = (
                        min(highVolume[store_idx], lowVolume[store_idx])
                        / max_vol
                        * 100.0
                    )
                else:
                    percentage[store_idx] = 100.0
                active[n_active] = store_idx
                n_active += 1

    return (
        ob, top_arr, bottom_arr, obVolume, lowVolume, highVolume,
        percentage, mitigated_index, breaker,
    )


@njit(cache=True)
def compute_zone_features_per_bar(
    n, close, high, low, atr,
    zone_creation, zone_low, zone_high, zone_type,
    zone_strength, zone_tf_weight, zone_freshness,
    zone_lookback, zone_proximity_atr,
):
    """
    Per-bar zone proximity, containment, and stack features.

    zone_creation must be sorted ascending.
    zone_freshness is mutated in-place (0=UNTESTED, 1=TESTED_ONCE, 2=MITIGATED).

    Returns 12 arrays (all length n):
      nearest_demand_dist, nearest_demand_strength, nearest_demand_freshness,
      nearest_demand_tf_weight,
      nearest_supply_dist, nearest_supply_strength, nearest_supply_freshness,
      nearest_supply_tf_weight,
      in_demand_zone, in_supply_zone, zone_stack_long, zone_stack_short
    """
    n_zones = len(zone_creation)

    nearest_demand_dist = np.full(n, np.nan)
    nearest_demand_strength = np.full(n, np.nan)
    nearest_demand_freshness = np.full(n, np.nan)
    nearest_demand_tf_weight = np.full(n, np.nan)
    nearest_supply_dist = np.full(n, np.nan)
    nearest_supply_strength = np.full(n, np.nan)
    nearest_supply_freshness = np.full(n, np.nan)
    nearest_supply_tf_weight = np.full(n, np.nan)
    in_demand_zone = np.zeros(n, dtype=np.int32)
    in_supply_zone = np.zeros(n, dtype=np.int32)
    zone_stack_long = np.zeros(n, dtype=np.int32)
    zone_stack_short = np.zeros(n, dtype=np.int32)

    for i in range(n):
        atr_val = atr[i]

        # --- Window: zone_creation in [i - zone_lookback, i] ---
        lo_bound = i - zone_lookback
        if lo_bound < 0:
            lo_bound = 0
        win_start = np.searchsorted(zone_creation, lo_bound)
        win_end = np.searchsorted(zone_creation, i, side='right')

        if win_start >= win_end:
            continue
        if atr_val == 0.0:
            # Can still update freshness, but skip distance/proximity
            for z in range(win_start, win_end):
                if zone_freshness[z] < 2:
                    if zone_low[z] <= close[i] <= zone_high[z]:
                        zone_freshness[z] += 1
            continue

        # --- Update freshness (zone_recent & freshness < 2) ---
        for z in range(win_start, win_end):
            if zone_freshness[z] < 2:
                if zone_low[z] <= close[i] <= zone_high[z]:
                    zone_freshness[z] += 1

        # --- Nearest demand/supply (ALL zone_recent, including mitigated) ---
        best_demand_dist = np.inf
        best_demand_idx = -1
        best_supply_dist = np.inf
        best_supply_idx = -1

        for z in range(win_start, win_end):
            mid = (zone_low[z] + zone_high[z]) * 0.5
            dist = abs(close[i] - mid) / atr_val

            if zone_type[z] == 1:  # demand
                if dist < best_demand_dist:
                    best_demand_dist = dist
                    best_demand_idx = z
            else:  # supply (type == -1)
                if dist < best_supply_dist:
                    best_supply_dist = dist
                    best_supply_idx = z

        if best_demand_idx >= 0:
            nearest_demand_dist[i] = best_demand_dist
            nearest_demand_strength[i] = zone_strength[best_demand_idx]
            nearest_demand_freshness[i] = float(zone_freshness[best_demand_idx])
            nearest_demand_tf_weight[i] = float(zone_tf_weight[best_demand_idx])

        if best_supply_idx >= 0:
            nearest_supply_dist[i] = best_supply_dist
            nearest_supply_strength[i] = zone_strength[best_supply_idx]
            nearest_supply_freshness[i] = float(zone_freshness[best_supply_idx])
            nearest_supply_tf_weight[i] = float(zone_tf_weight[best_supply_idx])

        # --- in_zone and stack (zone_recent & freshness < 2 only) ---
        for z in range(win_start, win_end):
            if zone_freshness[z] >= 2:
                continue

            mid = (zone_low[z] + zone_high[z]) * 0.5

            if zone_type[z] == 1:  # demand
                if zone_low[z] <= close[i] <= zone_high[z]:
                    in_demand_zone[i] = 1
                d = (close[i] - mid) / atr_val
                if d > 0.0 and d <= zone_proximity_atr:
                    zone_stack_long[i] += 1
            else:  # supply
                if zone_low[z] <= close[i] <= zone_high[z]:
                    in_supply_zone[i] = 1
                d = (mid - close[i]) / atr_val
                if d > 0.0 and d <= zone_proximity_atr:
                    zone_stack_short[i] += 1

    return (
        nearest_demand_dist, nearest_demand_strength,
        nearest_demand_freshness, nearest_demand_tf_weight,
        nearest_supply_dist, nearest_supply_strength,
        nearest_supply_freshness, nearest_supply_tf_weight,
        in_demand_zone, in_supply_zone,
        zone_stack_long, zone_stack_short,
    )
