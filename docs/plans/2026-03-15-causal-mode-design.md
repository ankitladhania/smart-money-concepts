# Causal Mode (`causal=True`) Design

## Problem

The library has lookahead bias in 6 of 8 functions. `swing_highs_lows` is the root cause (uses `shift(-N)` to look forward), and contaminates all dependent functions (`bos_choch`, `ob`, `liquidity`, `retracements`). `fvg` independently uses `shift(-1)`. Only `previous_high_low` and `sessions` are already causal.

This makes the library unsuitable for live trading or unbiased backtesting without modification.

## Solution

Add `causal: bool = False` parameter to 6 affected functions. When `causal=True`, all computations use only past/present data. Default `False` preserves backward compatibility.

## Approach

Full causal reimplementation (not compute-then-mask). Each function's causal code path genuinely only uses lookback operations.

## API Design

### Parameter
- `causal: bool = False` added to: `swing_highs_lows`, `fvg`, `bos_choch`, `ob`, `liquidity`, `retracements`
- No changes to `previous_high_low`, `sessions` (already causal)

### Metadata
- Returned DataFrames set `df.attrs['causal'] = True` when computed in causal mode

### Validation
- Dependent functions (`bos_choch`, `ob`, `liquidity`, `retracements`) check `swing_highs_lows.attrs.get('causal', False)` when called with `causal=True`
- Raise `ValueError` if input swings aren't causal

## Function-level Design

### `swing_highs_lows(ohlc, swing_length=50, causal=False)`

**Current**: Centered window via `shift(-(swing_length))` + `rolling(swing_length*2).max()`. Looks forward `swing_length` bars.

**Causal**:
- Lookback-only `rolling(2 * swing_length).max()`
- At bar `j`, confirms swing at `p = j - swing_length`: checks `high[p] == max(high[p-swing_length : j+1])`
- Uses positive shifts and lookback rolling only
- Last `swing_length` bars are NaN (unconfirmable)
- No synthetic swing point at end of data (current code forces one at `[-1]`)
- Synthetic point at `[0]` kept (initialization)
- Same deduplication on confirmed bars

### `fvg(ohlc, join_consecutive=False, causal=False)`

**Current**: `shift(-1)` reads next bar's low/high.

**Causal**:
- FVG at bar `i` confirmed at bar `i+1`
- At bar `j`: `high[j-2] < low[j]` (bullish) or `low[j-2] > high[j]` (bearish), direction from bar `j-1`
- Label placed at `j-1`
- Last bar is NaN
- Mitigation: bar-by-bar active FVG tracking. `MitigatedIndex` = NaN until mitigated

### `bos_choch(ohlc, swing_highs_lows, close_break=True, causal=False)`

**Current**: Pattern detection is causal, but break validation searches forward. Removes unbroken patterns.

**Causal**:
- Pattern detection unchanged
- Break validation: bar-by-bar with active pattern list
- `BrokenIndex` = NaN for not-yet-broken patterns (instead of deleting)
- Supersession logic preserved

### `ob(ohlc, swing_highs_lows, close_mitigation=False, causal=False)`

Already mostly causal (bar-by-bar with active OB lists). Only needs `causal` parameter + input validation.

### `liquidity(ohlc, swing_highs_lows, range_percent=0.01, causal=False)`

**Current**: Grouping is causal, sweep searches forward.

**Causal**:
- Grouping unchanged
- Sweep: bar-by-bar tracking. `Swept` = NaN until actually swept

### `retracements(ohlc, swing_highs_lows, causal=False)`

Already causal once swings are provided. Only needs `causal` parameter + input validation.

## Testing Strategy (TDD)

Tests written before implementation. For each function:

1. **No lookahead**: causal output at bar `i` computable from bars `0..i+confirmation_delay` only (verified by running on progressively truncated data)
2. **Equivalence on confirmed bars**: causal output (excluding tail NaNs) matches non-causal output
3. **Tail NaN correctness**: last N bars are NaN where N = confirmation window
4. **Metadata**: `attrs['causal']` set on output
5. **Validation**: dependent functions raise `ValueError` with non-causal swings + `causal=True`
6. **Mitigation/sweep incrementality**: patterns mitigated only when price reaches them

## Label Placement

Labels are placed at the original bar where the event occurred, not at the confirmation bar. The last N bars (where N = confirmation window) are NaN because they cannot yet be confirmed. This matches standard TA library conventions.

## Two Types of Lookahead Eliminated

1. **Structural**: pattern identification no longer uses future data
2. **Mitigation/sweep**: forward-searching replaced with bar-by-bar incremental tracking
