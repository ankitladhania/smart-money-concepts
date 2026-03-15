# Causal Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `causal=True` parameter to eliminate lookahead bias from all SMC indicators.

**Architecture:** Full causal reimplementation behind a `causal: bool = False` flag on 6 functions. Lookback-only rolling windows for swing detection, bar-by-bar mitigation/sweep tracking, metadata validation between dependent functions. TDD throughout.

**Tech Stack:** Python, pandas, numpy, unittest

**Design doc:** `docs/plans/2026-03-15-causal-mode-design.md`

---

### Task 1: Write causal tests for `swing_highs_lows`

**Files:**
- Create: `tests/test_causal.py`

**Step 1: Write the failing tests**

```python
import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.smc import smc

# shared test data
test_instrument = "EURUSD"
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", test_instrument)
df = pd.read_csv(os.path.join(TEST_DATA_DIR, "EURUSD_15M.csv"))
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)


class TestSwingHighsLowsCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        """attrs['causal'] must be True on output."""
        result = smc.swing_highs_lows(df, swing_length=5, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_non_causal_metadata_not_set(self):
        """Default (causal=False) must NOT set attrs['causal']."""
        result = smc.swing_highs_lows(df, swing_length=5)
        self.assertFalse(result.attrs.get("causal", False))

    def test_tail_nan(self):
        """Last swing_length bars must be NaN in causal mode."""
        swing_length = 5
        result = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        tail = result["HighLow"].iloc[-swing_length:]
        self.assertTrue(tail.isna().all(),
                        f"Expected last {swing_length} bars to be NaN, got {tail.values}")

    def test_equivalence_on_confirmed_bars(self):
        """Causal output (minus tail) must match non-causal for confirmed bars."""
        swing_length = 5
        causal = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        non_causal = smc.swing_highs_lows(df, swing_length=swing_length, causal=False)

        # Exclude tail (last swing_length bars) and first bar (synthetic point differs)
        confirmed = len(df) - swing_length
        # Compare HighLow column for rows where causal has a value
        c = causal["HighLow"].iloc[1:confirmed]
        nc = non_causal["HighLow"].iloc[1:confirmed]
        # Every causal swing should exist in non-causal (causal is a subset
        # because end-of-data synthetic point and dedup edge cases may differ)
        causal_swings = c.dropna()
        for idx in causal_swings.index:
            self.assertEqual(
                causal_swings.loc[idx], nc.loc[idx],
                f"Mismatch at index {idx}: causal={causal_swings.loc[idx]}, non_causal={nc.loc[idx]}"
            )

    def test_no_lookahead_via_truncation(self):
        """Causal result at bar i must not change when future bars are removed."""
        swing_length = 5
        n = len(df)
        cutoff = n - 100  # remove last 100 bars

        full_result = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        truncated_result = smc.swing_highs_lows(
            df.iloc[:cutoff], swing_length=swing_length, causal=True
        )

        # Confirmed bars in truncated = cutoff - swing_length
        confirmed = cutoff - swing_length
        full_confirmed = full_result["HighLow"].iloc[:confirmed]
        trunc_confirmed = truncated_result["HighLow"].iloc[:confirmed]

        pd.testing.assert_series_equal(
            full_confirmed.reset_index(drop=True),
            trunc_confirmed.reset_index(drop=True),
            check_names=False,
            obj="Causal swing_highs_lows should not change when future data is removed",
        )

    def test_no_synthetic_endpoint(self):
        """Causal mode must NOT force a synthetic swing at the last bar."""
        swing_length = 5
        result = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        self.assertTrue(np.isnan(result["HighLow"].iloc[-1]),
                        "Last bar should be NaN in causal mode (no synthetic endpoint)")

    def test_existing_non_causal_unchanged(self):
        """Existing behavior must be preserved when causal=False."""
        result = smc.swing_highs_lows(df, swing_length=5, causal=False)
        expected = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "swing_highs_lows_result_data.csv")
        )
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_causal.py::TestSwingHighsLowsCausal -v`
Expected: FAIL (TypeError: unexpected keyword argument 'causal')

**Step 3: Commit test file**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for swing_highs_lows"
```

---

### Task 2: Implement causal `swing_highs_lows`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:137-219`

**Step 1: Add causal parameter and implementation**

Add `causal: bool = False` to the signature. When `causal=True`:

```python
@classmethod
def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50, causal: bool = False) -> Series:
    if not causal:
        # ... existing code unchanged ...
        pass
    else:
        # Causal implementation
        n = len(ohlc)
        high = ohlc["high"].values
        low = ohlc["low"].values
        half = swing_length
        full_window = 2 * half

        # Lookback-only rolling max/min (window = 2*swing_length)
        rolling_max = pd.Series(high).rolling(full_window, min_periods=full_window).max().values
        rolling_min = pd.Series(low).rolling(full_window, min_periods=full_window).min().values

        # rolling_max[j] = max(high[j-full_window+1 : j+1])
        # Swing at p = j - half confirmed at j
        # Check: high[p] == rolling_max[j] where j = p + half
        swing_highs_lows = np.full(n, np.nan)

        for j in range(full_window - 1, n):
            p = j - half
            if p < 0:
                continue
            if high[p] == rolling_max[j]:
                swing_highs_lows[p] = 1
            elif low[p] == rolling_min[j]:
                swing_highs_lows[p] = -1

        # Last half bars are NaN (unconfirmed) - already NaN by default

        # Deduplication (same as non-causal)
        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]
            if len(positions) < 2:
                break
            current = swing_highs_lows[positions[:-1]]
            next_vals = swing_highs_lows[positions[1:]]
            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values
            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values
            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next_vals == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            consecutive_lows = (current == -1) & (next_vals == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            if not index_to_remove.any():
                break
            swing_highs_lows[positions[index_to_remove]] = np.nan

        # Synthetic start point only (no endpoint in causal mode)
        positions = np.where(~np.isnan(swing_highs_lows))[0]
        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            elif swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        result = pd.concat(
            [pd.Series(swing_highs_lows, name="HighLow"),
             pd.Series(level, name="Level")],
            axis=1,
        )
        result.attrs["causal"] = True
        return result
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_causal.py::TestSwingHighsLowsCausal -v`
Expected: All PASS

**Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/unit_tests.py -v`
Expected: All 10 PASS

**Step 4: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to swing_highs_lows"
```

---

### Task 3: Write causal tests for `fvg`

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add FVG causal tests**

```python
class TestFVGCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        result = smc.fvg(df, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_non_causal_metadata_not_set(self):
        result = smc.fvg(df)
        self.assertFalse(result.attrs.get("causal", False))

    def test_tail_nan(self):
        """Last 1 bar must be NaN (needs next bar to confirm)."""
        result = smc.fvg(df, causal=True)
        self.assertTrue(np.isnan(result["FVG"].iloc[-1]))

    def test_equivalence_on_confirmed_bars(self):
        """Causal FVG detection should match non-causal for confirmed bars."""
        causal = smc.fvg(df, causal=True)
        non_causal = smc.fvg(df, causal=False)
        # Compare all but last bar
        confirmed = len(df) - 1
        c_fvg = causal["FVG"].iloc[:confirmed]
        nc_fvg = non_causal["FVG"].iloc[:confirmed]
        # Where causal detects an FVG, non-causal should agree
        causal_fvgs = c_fvg.dropna()
        for idx in causal_fvgs.index:
            self.assertEqual(causal_fvgs.loc[idx], nc_fvg.loc[idx],
                             f"FVG mismatch at {idx}")

    def test_no_lookahead_via_truncation(self):
        """Causal FVG at bar i must not change when future bars removed."""
        n = len(df)
        cutoff = n - 100

        full = smc.fvg(df, causal=True)
        trunc = smc.fvg(df.iloc[:cutoff], causal=True)

        confirmed = cutoff - 1
        pd.testing.assert_series_equal(
            full["FVG"].iloc[:confirmed].reset_index(drop=True),
            trunc["FVG"].iloc[:confirmed].reset_index(drop=True),
            check_names=False,
        )

    def test_mitigation_causal(self):
        """MitigatedIndex must not reference bars beyond what's available."""
        result = smc.fvg(df, causal=True)
        fvg_rows = result[result["FVG"].notna()]
        for idx in fvg_rows.index:
            mit_idx = result["MitigatedIndex"].iloc[idx]
            if not np.isnan(mit_idx) and mit_idx != 0:
                # Mitigation index must be after the FVG
                self.assertGreater(int(mit_idx), idx,
                                   f"Mitigation at {mit_idx} must be after FVG at {idx}")

    def test_mitigation_no_lookahead_via_truncation(self):
        """Mitigation found in truncated data must match full data."""
        n = len(df)
        cutoff = n - 100

        full = smc.fvg(df, causal=True)
        trunc = smc.fvg(df.iloc[:cutoff], causal=True)

        # For FVGs in truncated data that are mitigated, the mitigation index
        # must be the same in both
        trunc_fvgs = trunc[trunc["FVG"].notna() & (trunc["MitigatedIndex"] > 0)]
        for idx in trunc_fvgs.index:
            self.assertEqual(
                trunc["MitigatedIndex"].iloc[idx],
                full["MitigatedIndex"].iloc[idx],
                f"Mitigation mismatch at FVG index {idx}",
            )

    def test_existing_non_causal_unchanged(self):
        result = smc.fvg(df, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "fvg_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_causal.py::TestFVGCausal -v`
Expected: FAIL

**Step 3: Commit**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for fvg"
```

---

### Task 4: Implement causal `fvg`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:56-134`

**Step 1: Add causal parameter and implementation**

Add `causal: bool = False` to signature. When `causal=True`:

- Detect FVGs using lookback only: at bar `j`, check `high[j-2] < low[j]` (bullish) / `low[j-2] > high[j]` (bearish), direction from bar `j-1`. Place at `j-1`.
- Bar-by-bar mitigation: maintain list of active FVGs, check each new bar against them.
- `join_consecutive` works the same on the causal results.
- Set `result.attrs["causal"] = True`.

**Step 2: Run tests**

Run: `python -m pytest tests/test_causal.py::TestFVGCausal tests/unit_tests.py::TestSmartMoneyConcepts::test_fvg tests/unit_tests.py::TestSmartMoneyConcepts::test_fvg_consecutive -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to fvg"
```

---

### Task 5: Write causal tests for `bos_choch`

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add bos_choch causal tests**

```python
class TestBosChochCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.bos_choch(df, swings, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_rejects_non_causal_swings(self):
        """Must raise ValueError when causal=True but swings are non-causal."""
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        with self.assertRaises(ValueError):
            smc.bos_choch(df, swings, causal=True)

    def test_no_lookahead_via_truncation(self):
        """BOS/CHOCH at bar i must not change when future bars removed."""
        cutoff = len(df) - 200

        full_swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        full = smc.bos_choch(df, full_swings, causal=True)

        trunc_swings = smc.swing_highs_lows(df.iloc[:cutoff], swing_length=5, causal=True)
        trunc = smc.bos_choch(df.iloc[:cutoff], trunc_swings, causal=True)

        # Confirmed region: exclude last swing_length bars from truncated
        confirmed = cutoff - 50  # generous margin
        for col in ["BOS", "CHOCH"]:
            full_vals = full[col].iloc[:confirmed]
            trunc_vals = trunc[col].iloc[:confirmed]
            # Where truncated has a value, full must agree
            trunc_non_nan = trunc_vals.dropna()
            for idx in trunc_non_nan.index:
                self.assertEqual(
                    trunc_non_nan.loc[idx], full_vals.loc[idx],
                    f"{col} mismatch at {idx}",
                )

    def test_broken_index_causal(self):
        """BrokenIndex must be after the BOS/CHOCH bar, or NaN if not yet broken."""
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.bos_choch(df, swings, causal=True)
        for col in ["BOS", "CHOCH"]:
            events = result[result[col].notna()]
            for idx in events.index:
                broken = result["BrokenIndex"].iloc[idx]
                if not np.isnan(broken):
                    self.assertGreater(int(broken), idx)

    def test_existing_non_causal_unchanged(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        result = smc.bos_choch(df, swings, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "bos_choch_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
```

**Step 2: Run to verify failure, then commit**

Run: `python -m pytest tests/test_causal.py::TestBosChochCausal -v`

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for bos_choch"
```

---

### Task 6: Implement causal `bos_choch`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:222-373`

**Step 1: Add causal parameter**

- Add `causal: bool = False` to signature
- When `causal=True`: validate `swing_highs_lows.attrs.get('causal', False)`, raise `ValueError` if not
- Pattern detection loop unchanged (already causal)
- Replace forward break-search with bar-by-bar: iterate through all bars, check active patterns
- Keep unbroken patterns (BrokenIndex = NaN) instead of removing them
- Set `result.attrs["causal"] = True`

**Step 2: Run tests**

Run: `python -m pytest tests/test_causal.py::TestBosChochCausal tests/unit_tests.py::TestSmartMoneyConcepts::test_bos_choch -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to bos_choch"
```

---

### Task 7: Write causal tests for `ob`

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add OB causal tests**

```python
class TestOBCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.ob(df, swings, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_rejects_non_causal_swings(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        with self.assertRaises(ValueError):
            smc.ob(df, swings, causal=True)

    def test_mitigation_index_causal(self):
        """MitigatedIndex must be after the OB bar."""
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.ob(df, swings, causal=True)
        obs = result[result["OB"].notna()]
        for idx in obs.index:
            mit = result["MitigatedIndex"].iloc[idx]
            if not np.isnan(mit) and mit != 0:
                self.assertGreater(int(mit), idx)

    def test_existing_non_causal_unchanged(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        result = smc.ob(df, swings, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "ob_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
```

**Step 2: Run to verify failure, then commit**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for ob"
```

---

### Task 8: Implement causal `ob`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:376-570`

**Step 1: Add causal parameter**

- Add `causal: bool = False` to signature
- Validate causal swings input
- Core OB logic is already bar-by-bar — minimal changes needed
- Set `result.attrs["causal"] = True`

**Step 2: Run tests**

Run: `python -m pytest tests/test_causal.py::TestOBCausal tests/unit_tests.py::TestSmartMoneyConcepts::test_ob -v`

**Step 3: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to ob"
```

---

### Task 9: Write causal tests for `liquidity`

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add liquidity causal tests**

```python
class TestLiquidityCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.liquidity(df, swings, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_rejects_non_causal_swings(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        with self.assertRaises(ValueError):
            smc.liquidity(df, swings, causal=True)

    def test_swept_causal(self):
        """Swept index must be after the liquidity bar, or NaN/0 if not swept."""
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.liquidity(df, swings, causal=True)
        liqs = result[result["Liquidity"].notna()]
        for idx in liqs.index:
            swept = result["Swept"].iloc[idx]
            if not np.isnan(swept) and swept != 0:
                self.assertGreater(int(swept), idx)

    def test_no_lookahead_via_truncation(self):
        """Liquidity grouping must not change when future bars removed."""
        cutoff = len(df) - 200
        full_swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        full = smc.liquidity(df, full_swings, causal=True)
        trunc_swings = smc.swing_highs_lows(df.iloc[:cutoff], swing_length=5, causal=True)
        trunc = smc.liquidity(df.iloc[:cutoff], trunc_swings, causal=True)

        confirmed = cutoff - 50
        trunc_liqs = trunc["Liquidity"].iloc[:confirmed].dropna()
        for idx in trunc_liqs.index:
            self.assertEqual(
                trunc_liqs.loc[idx], full["Liquidity"].iloc[idx],
                f"Liquidity mismatch at {idx}",
            )

    def test_existing_non_causal_unchanged(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        result = smc.liquidity(df, swings, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "liquidity_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
```

**Step 2: Run to verify failure, then commit**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for liquidity"
```

---

### Task 10: Implement causal `liquidity`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:573-698`

**Step 1: Add causal parameter**

- Add `causal: bool = False` to signature
- Validate causal swings input
- Replace forward sweep search with bar-by-bar: track active liquidity zones, check each bar for sweep
- `Swept` = NaN until actually swept
- Grouping no longer bounded by forward sweep — group candidates as they appear
- Set `result.attrs["causal"] = True`

**Step 2: Run tests**

Run: `python -m pytest tests/test_causal.py::TestLiquidityCausal tests/unit_tests.py::TestSmartMoneyConcepts::test_liquidity -v`

**Step 3: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to liquidity"
```

---

### Task 11: Write causal tests for `retracements`

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add retracements causal tests**

```python
class TestRetracementsCausal(unittest.TestCase):

    def test_causal_metadata_set(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        result = smc.retracements(df, swings, causal=True)
        self.assertTrue(result.attrs.get("causal", False))

    def test_rejects_non_causal_swings(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        with self.assertRaises(ValueError):
            smc.retracements(df, swings, causal=True)

    def test_existing_non_causal_unchanged(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        result = smc.retracements(df, swings, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "retracements_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
```

**Step 2: Run to verify failure, then commit**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode tests for retracements"
```

---

### Task 12: Implement causal `retracements`

**Files:**
- Modify: `smartmoneyconcepts/smc.py:900-985`

**Step 1: Add causal parameter**

- Add `causal: bool = False` to signature
- Validate causal swings input
- Core logic already causal — no changes to computation
- Set `result.attrs["causal"] = True`

**Step 2: Run all tests**

Run: `python -m pytest tests/test_causal.py tests/unit_tests.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add smartmoneyconcepts/smc.py
git commit -m "feat: add causal mode to retracements"
```

---

### Task 13: Final integration test

**Files:**
- Modify: `tests/test_causal.py`

**Step 1: Add end-to-end integration test**

```python
class TestCausalIntegration(unittest.TestCase):

    def test_full_pipeline_causal(self):
        """Run all functions in causal mode end-to-end."""
        swings = smc.swing_highs_lows(df, swing_length=5, causal=True)
        self.assertTrue(swings.attrs.get("causal"))

        fvg = smc.fvg(df, causal=True)
        self.assertTrue(fvg.attrs.get("causal"))

        bos_choch = smc.bos_choch(df, swings, causal=True)
        self.assertTrue(bos_choch.attrs.get("causal"))

        ob = smc.ob(df, swings, causal=True)
        self.assertTrue(ob.attrs.get("causal"))

        liq = smc.liquidity(df, swings, causal=True)
        self.assertTrue(liq.attrs.get("causal"))

        ret = smc.retracements(df, swings, causal=True)
        self.assertTrue(ret.attrs.get("causal"))

    def test_all_existing_tests_still_pass(self):
        """Sanity: default causal=False produces same results as before."""
        # This is covered by individual test_existing_non_causal_unchanged tests
        # but this confirms the full pipeline
        swings = smc.swing_highs_lows(df, swing_length=5)
        self.assertFalse(swings.attrs.get("causal", False))
        bos = smc.bos_choch(df, swings)
        self.assertFalse(bos.attrs.get("causal", False))
```

**Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Final commit**

```bash
git add tests/test_causal.py
git commit -m "test: add causal mode integration tests"
```
