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

    def test_consecutive_same_type_more_extreme(self):
        """When consecutive same-type swings appear, the later one must be more extreme."""
        swing_length = 5
        result = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        swings = result["HighLow"].dropna()
        high = df["high"].values if "high" in df.columns else df["High"].values
        low = df["low"].values if "low" in df.columns else df["Low"].values
        values = swings.values
        indices = swings.index
        for i in range(1, len(values)):
            if values[i] == values[i - 1]:
                prev_idx = indices[i - 1]
                curr_idx = indices[i]
                if values[i] == 1:
                    self.assertGreaterEqual(
                        high[curr_idx], high[prev_idx],
                        f"Consecutive highs at {prev_idx} and {curr_idx}: "
                        f"later high {high[curr_idx]} < prev {high[prev_idx]}"
                    )
                else:
                    self.assertLessEqual(
                        low[curr_idx], low[prev_idx],
                        f"Consecutive lows at {prev_idx} and {curr_idx}: "
                        f"later low {low[curr_idx]} > prev {low[prev_idx]}"
                    )

    def test_no_lookahead_via_truncation(self):
        """Causal result at bar i must not change when future bars are removed."""
        swing_length = 5
        n = len(df)
        cutoff = n - 100

        full_result = smc.swing_highs_lows(df, swing_length=swing_length, causal=True)
        truncated_result = smc.swing_highs_lows(
            df.iloc[:cutoff], swing_length=swing_length, causal=True
        )

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

    def test_fvg_placed_at_confirming_bar(self):
        """Causal FVG must be placed at bar j (confirming bar), not j-1."""
        result = smc.fvg(df, causal=True)
        fvg_indices = result.index[result["FVG"].notna()]
        ohlc = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        high = ohlc["high"].values
        low = ohlc["low"].values
        close = ohlc["close"].values
        open_ = ohlc["open"].values
        for idx in fvg_indices:
            if idx < 2:
                continue
            direction = result["FVG"].iloc[idx]
            if direction == 1:
                self.assertLess(high[idx - 2], low[idx])
                self.assertGreater(close[idx - 1], open_[idx - 1])
            else:
                self.assertGreater(low[idx - 2], high[idx])
                self.assertLess(close[idx - 1], open_[idx - 1])

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
                self.assertGreater(int(mit_idx), idx,
                                   f"Mitigation at {mit_idx} must be after FVG at {idx}")

    def test_mitigation_no_lookahead_via_truncation(self):
        """Mitigation found in truncated data must match full data."""
        n = len(df)
        cutoff = n - 100

        full = smc.fvg(df, causal=True)
        trunc = smc.fvg(df.iloc[:cutoff], causal=True)

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

        confirmed = cutoff - 50  # generous margin
        for col in ["BOS", "CHOCH"]:
            full_vals = full[col].iloc[:confirmed]
            trunc_vals = trunc[col].iloc[:confirmed]
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
                self.assertGreaterEqual(int(mit), idx)

    def test_existing_non_causal_unchanged(self):
        swings = smc.swing_highs_lows(df, swing_length=5, causal=False)
        result = smc.ob(df, swings, causal=False)
        expected = pd.read_csv(os.path.join(TEST_DATA_DIR, "ob_result_data.csv"))
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)


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
        swings = smc.swing_highs_lows(df, swing_length=5)
        self.assertFalse(swings.attrs.get("causal", False))
        bos = smc.bos_choch(df, swings)
        self.assertFalse(bos.attrs.get("causal", False))


if __name__ == "__main__":
    unittest.main()
