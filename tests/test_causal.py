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

        confirmed = len(df) - swing_length
        c = causal["HighLow"].iloc[1:confirmed]
        nc = non_causal["HighLow"].iloc[1:confirmed]
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


if __name__ == "__main__":
    unittest.main()
