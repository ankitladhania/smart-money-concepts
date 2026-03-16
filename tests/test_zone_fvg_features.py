import numpy as np
import unittest

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestComputeZoneFeaturesPerBar(unittest.TestCase):
    """Tests for compute_zone_features_per_bar njit helper."""

    def _make_bars(self, n, close, high=None, low=None, atr=None):
        """Helper to build bar arrays with sensible defaults."""
        close = np.array(close, dtype=np.float64)
        if high is None:
            high = close + 1.0
        else:
            high = np.array(high, dtype=np.float64)
        if low is None:
            low = close - 1.0
        else:
            low = np.array(low, dtype=np.float64)
        if atr is None:
            atr = np.ones(n, dtype=np.float64)
        else:
            atr = np.array(atr, dtype=np.float64)
        return close, high, low, atr

    def test_no_zones_returns_nan_and_zero(self):
        """With zero zones, all distances are NaN, all flags/counts are 0."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 5
        close, high, low, atr = self._make_bars(n, [100.0] * n)
        # Empty zone arrays
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int32)

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            empty_i,   # zone_creation
            empty_f,   # zone_low
            empty_f,   # zone_high
            empty_i,   # zone_type
            empty_f,   # zone_strength
            empty_i,   # zone_tf_weight
            empty_i,   # zone_freshness
            100,       # zone_lookback
            3.0,       # zone_proximity_atr
        )
        # 12 output arrays
        self.assertEqual(len(result), 12)
        # Distances are NaN
        for arr in result[:8]:
            self.assertTrue(np.all(np.isnan(arr)), f"Expected all NaN, got {arr}")
        # Flags/counts are 0
        for arr in result[8:]:
            self.assertTrue(np.all(arr == 0), f"Expected all 0, got {arr}")

    def test_nearest_demand_and_supply(self):
        """Nearest zone picked by min abs(close - midpoint) / atr."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 5
        close, high, low, atr = self._make_bars(n, [100.0] * n)

        # Two zones: demand at 95-97 (mid=96), supply at 103-105 (mid=104)
        zone_creation = np.array([0, 0], dtype=np.int32)
        zone_low       = np.array([95.0, 103.0], dtype=np.float64)
        zone_high      = np.array([97.0, 105.0], dtype=np.float64)
        zone_type      = np.array([1, -1], dtype=np.int32)  # demand, supply
        zone_strength  = np.array([80.0, 60.0], dtype=np.float64)
        zone_tf_weight = np.array([3, 5], dtype=np.int32)
        zone_freshness = np.array([0, 0], dtype=np.int32)

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 10.0,
        )
        (nd_dist, nd_str, nd_fresh, nd_tf,
         ns_dist, ns_str, ns_fresh, ns_tf,
         in_demand, in_supply, stack_long, stack_short) = result

        # Demand mid=96, close=100, dist=4.0/1.0=4.0
        np.testing.assert_allclose(nd_dist, 4.0)
        np.testing.assert_allclose(nd_str, 80.0)
        np.testing.assert_allclose(nd_tf, 3.0)
        # Supply mid=104, close=100, dist=4.0/1.0=4.0
        np.testing.assert_allclose(ns_dist, 4.0)
        np.testing.assert_allclose(ns_str, 60.0)
        np.testing.assert_allclose(ns_tf, 5.0)

    def test_freshness_state_machine(self):
        """Freshness transitions: 0 -> 1 -> 2 (frozen) based on close inside zone."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        # 5 bars: bars 0,1 close outside zone, bars 2,3 close inside, bar 4 inside (but frozen)
        n = 5
        close = np.array([90.0, 90.0, 100.0, 100.0, 100.0], dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)

        # Zone at 99-101 (close=100 is inside), created at bar 0
        zone_creation  = np.array([0], dtype=np.int32)
        zone_low       = np.array([99.0], dtype=np.float64)
        zone_high      = np.array([101.0], dtype=np.float64)
        zone_type      = np.array([1], dtype=np.int32)
        zone_strength  = np.array([70.0], dtype=np.float64)
        zone_tf_weight = np.array([2], dtype=np.int32)
        zone_freshness = np.array([0], dtype=np.int32)

        compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 10.0,
        )
        # After bar 0,1: close=90 outside [99,101] -> stays 0
        # After bar 2: close=100 inside -> 1
        # After bar 3: close=100 inside -> 2
        # After bar 4: close=100 inside but frozen at 2
        self.assertEqual(zone_freshness[0], 2)

    def test_freshness_increments_per_bar(self):
        """Freshness goes 0->1 on first inside bar, 1->2 on second, stays 2."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 4
        # bar0: outside, bar1: inside, bar2: outside, bar3: inside
        close = np.array([90.0, 100.0, 90.0, 100.0], dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)

        zone_creation  = np.array([0], dtype=np.int32)
        zone_low       = np.array([99.0], dtype=np.float64)
        zone_high      = np.array([101.0], dtype=np.float64)
        zone_type      = np.array([1], dtype=np.int32)
        zone_strength  = np.array([70.0], dtype=np.float64)
        zone_tf_weight = np.array([2], dtype=np.int32)
        zone_freshness = np.array([0], dtype=np.int32)

        compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 10.0,
        )
        # bar1 inside -> 0->1, bar3 inside -> 1->2
        self.assertEqual(zone_freshness[0], 2)

    def test_in_zone_only_active(self):
        """in_demand_zone/in_supply_zone only count zones with freshness < 2."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 1
        close = np.array([100.0], dtype=np.float64)
        high = np.array([101.0], dtype=np.float64)
        low = np.array([99.0], dtype=np.float64)
        atr = np.ones(n, dtype=np.float64)

        # Demand zone at 99-101, already mitigated (freshness=2)
        zone_creation  = np.array([0], dtype=np.int32)
        zone_low       = np.array([99.0], dtype=np.float64)
        zone_high      = np.array([101.0], dtype=np.float64)
        zone_type      = np.array([1], dtype=np.int32)
        zone_strength  = np.array([70.0], dtype=np.float64)
        zone_tf_weight = np.array([2], dtype=np.int32)
        zone_freshness = np.array([2], dtype=np.int32)  # already mitigated

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 10.0,
        )
        in_demand = result[8]
        # Close is inside zone, but freshness=2 so NOT counted as in_demand
        self.assertEqual(in_demand[0], 0)
        # But nearest_demand_dist should still be computed (uses all recent zones)
        nd_dist = result[0]
        self.assertFalse(np.isnan(nd_dist[0]))

    def test_zone_stack_counts(self):
        """zone_stack_long counts demand below price, zone_stack_short supply above."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 1
        close = np.array([100.0], dtype=np.float64)
        high = np.array([101.0], dtype=np.float64)
        low = np.array([99.0], dtype=np.float64)
        atr = np.ones(n, dtype=np.float64)

        # 2 demand zones below (mid=91, mid=96), 1 supply above (mid=104)
        # 1 demand zone above (mid=106) — should NOT count in stack_long
        zone_creation  = np.array([0, 0, 0, 0], dtype=np.int32)
        zone_low       = np.array([90.0, 95.0, 103.0, 105.0], dtype=np.float64)
        zone_high      = np.array([92.0, 97.0, 105.0, 107.0], dtype=np.float64)
        zone_type      = np.array([1, 1, -1, 1], dtype=np.int32)
        zone_strength  = np.array([70.0, 80.0, 60.0, 50.0], dtype=np.float64)
        zone_tf_weight = np.array([2, 3, 4, 1], dtype=np.int32)
        zone_freshness = np.array([0, 0, 0, 0], dtype=np.int32)

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 20.0,  # large proximity to include all
        )
        stack_long = result[10]
        stack_short = result[11]
        # 2 demand zones with mid below price (91, 96), demand at 106 is above -> excluded
        self.assertEqual(stack_long[0], 2)
        # 1 supply zone with mid above price (104)
        self.assertEqual(stack_short[0], 1)

    def test_lookback_window(self):
        """Zones outside the lookback window are ignored."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 10
        close, high, low, atr = self._make_bars(n, [100.0] * n)

        # Zone created at bar 0, lookback=3 -> only visible at bars 0,1,2,3
        zone_creation  = np.array([0], dtype=np.int32)
        zone_low       = np.array([95.0], dtype=np.float64)
        zone_high      = np.array([97.0], dtype=np.float64)
        zone_type      = np.array([1], dtype=np.int32)
        zone_strength  = np.array([80.0], dtype=np.float64)
        zone_tf_weight = np.array([3], dtype=np.int32)
        zone_freshness = np.array([0], dtype=np.int32)

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            3, 10.0,  # lookback=3
        )
        nd_dist = result[0]
        # Bar 3: zone_creation=0, i=3, i-lookback=0, so 0 >= 0 -> in window
        self.assertFalse(np.isnan(nd_dist[3]))
        # Bar 4: zone_creation=0, i=4, i-lookback=1, so 0 < 1 -> out of window
        self.assertTrue(np.isnan(nd_dist[4]))

    def test_atr_zero_guard(self):
        """When atr=0, distances default to NaN, flags/counts still work."""
        from smartmoneyconcepts._numba_helpers import compute_zone_features_per_bar

        n = 1
        close = np.array([100.0], dtype=np.float64)
        high = np.array([101.0], dtype=np.float64)
        low = np.array([99.0], dtype=np.float64)
        atr = np.array([0.0], dtype=np.float64)

        zone_creation  = np.array([0], dtype=np.int32)
        zone_low       = np.array([99.0], dtype=np.float64)
        zone_high      = np.array([101.0], dtype=np.float64)
        zone_type      = np.array([1], dtype=np.int32)
        zone_strength  = np.array([70.0], dtype=np.float64)
        zone_tf_weight = np.array([2], dtype=np.int32)
        zone_freshness = np.array([0], dtype=np.int32)

        result = compute_zone_features_per_bar(
            n, close, high, low, atr,
            zone_creation, zone_low, zone_high, zone_type,
            zone_strength, zone_tf_weight, zone_freshness,
            100, 10.0,
        )
        nd_dist = result[0]
        self.assertTrue(np.isnan(nd_dist[0]))


class TestComputeFvgFeaturesPerBar(unittest.TestCase):
    """Tests for compute_fvg_features_per_bar njit helper."""

    def test_no_fvgs_returns_nan_and_zero(self):
        """With zero FVGs, distances are NaN, flags/counts are 0."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 5
        close = np.array([100.0] * n, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)
        empty_f = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int32)

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            empty_i, empty_f, empty_f, empty_i, empty_i,
            100, 3.0,
        )
        self.assertEqual(len(result), 5)
        for arr in result[:2]:
            self.assertTrue(np.all(np.isnan(arr)))
        for arr in result[2:]:
            self.assertTrue(np.all(arr == 0))

    def test_nearest_bull_and_bear_fvg(self):
        """Nearest FVG by min abs(close - midpoint) / atr."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 5
        close = np.array([100.0] * n, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)

        # Bull FVG at 94-96 (mid=95), Bear FVG at 104-106 (mid=105)
        fvg_creation = np.array([0, 0], dtype=np.int32)
        fvg_low      = np.array([94.0, 104.0], dtype=np.float64)
        fvg_high     = np.array([96.0, 106.0], dtype=np.float64)
        fvg_type     = np.array([1, -1], dtype=np.int32)
        fvg_mit      = np.array([0, 0], dtype=np.int32)  # unfilled

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            fvg_creation, fvg_low, fvg_high, fvg_type, fvg_mit,
            100, 10.0,
        )
        bull_dist, bear_dist, in_bull, in_bear, count = result

        np.testing.assert_allclose(bull_dist, 5.0)  # |100-95|/1
        np.testing.assert_allclose(bear_dist, 5.0)  # |100-105|/1

    def test_fvg_filled_excluded(self):
        """FVG with mitigation_index <= current bar is excluded."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 5
        close = np.array([100.0] * n, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)

        # Bull FVG created at bar 0, mitigated at bar 2
        fvg_creation = np.array([0], dtype=np.int32)
        fvg_low      = np.array([94.0], dtype=np.float64)
        fvg_high     = np.array([96.0], dtype=np.float64)
        fvg_type     = np.array([1], dtype=np.int32)
        fvg_mit      = np.array([2], dtype=np.int32)  # filled at bar 2

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            fvg_creation, fvg_low, fvg_high, fvg_type, fvg_mit,
            100, 10.0,
        )
        bull_dist = result[0]
        # Bars 0,1: FVG unfilled (mit=2 > bar), so visible
        self.assertFalse(np.isnan(bull_dist[0]))
        self.assertFalse(np.isnan(bull_dist[1]))
        # Bars 2,3,4: FVG filled (mit=2 <= bar), so invisible
        self.assertTrue(np.isnan(bull_dist[2]))
        self.assertTrue(np.isnan(bull_dist[3]))

    def test_in_fvg_wick_overlap(self):
        """in_bull_fvg uses wick overlap, not close-based."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 1
        # Close outside FVG, but wick overlaps
        close = np.array([100.0], dtype=np.float64)
        high = np.array([106.0], dtype=np.float64)   # wick reaches into bear FVG
        low = np.array([94.0], dtype=np.float64)      # wick reaches into bull FVG
        atr = np.ones(n, dtype=np.float64)

        # Bull FVG at 93-95, Bear FVG at 105-107
        fvg_creation = np.array([0, 0], dtype=np.int32)
        fvg_low      = np.array([93.0, 105.0], dtype=np.float64)
        fvg_high     = np.array([95.0, 107.0], dtype=np.float64)
        fvg_type     = np.array([1, -1], dtype=np.int32)
        fvg_mit      = np.array([0, 0], dtype=np.int32)

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            fvg_creation, fvg_low, fvg_high, fvg_type, fvg_mit,
            100, 10.0,
        )
        in_bull = result[2]
        in_bear = result[3]
        self.assertEqual(in_bull[0], 1)  # low=94 <= fvg_high=95 and high=106 >= fvg_low=93
        self.assertEqual(in_bear[0], 1)  # low=94 <= fvg_high=107 and high=106 >= fvg_low=105

    def test_fvg_count_nearby(self):
        """Counts unfilled FVGs within proximity (both directions)."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 1
        close = np.array([100.0], dtype=np.float64)
        high = np.array([101.0], dtype=np.float64)
        low = np.array([99.0], dtype=np.float64)
        atr = np.ones(n, dtype=np.float64)

        # 3 FVGs: bull at mid=95 (dist=5), bear at mid=103 (dist=3), bull at mid=120 (dist=20)
        fvg_creation = np.array([0, 0, 0], dtype=np.int32)
        fvg_low      = np.array([94.0, 102.0, 119.0], dtype=np.float64)
        fvg_high     = np.array([96.0, 104.0, 121.0], dtype=np.float64)
        fvg_type     = np.array([1, -1, 1], dtype=np.int32)
        fvg_mit      = np.array([0, 0, 0], dtype=np.int32)

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            fvg_creation, fvg_low, fvg_high, fvg_type, fvg_mit,
            100, 10.0,  # proximity=10 ATR
        )
        count = result[4]
        # mid=95 dist=5 (<=10), mid=103 dist=3 (<=10), mid=120 dist=20 (>10)
        self.assertEqual(count[0], 2)

    def test_fvg_lookback_window(self):
        """FVGs outside lookback window are ignored."""
        from smartmoneyconcepts._numba_helpers import compute_fvg_features_per_bar

        n = 10
        close = np.array([100.0] * n, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        atr = np.ones(n, dtype=np.float64)

        # FVG created at bar 0, lookback=3
        fvg_creation = np.array([0], dtype=np.int32)
        fvg_low      = np.array([94.0], dtype=np.float64)
        fvg_high     = np.array([96.0], dtype=np.float64)
        fvg_type     = np.array([1], dtype=np.int32)
        fvg_mit      = np.array([0], dtype=np.int32)

        result = compute_fvg_features_per_bar(
            n, close, high, low, atr,
            fvg_creation, fvg_low, fvg_high, fvg_type, fvg_mit,
            3, 10.0,
        )
        bull_dist = result[0]
        # Bar 3: creation=0, i-lookback=0, so 0 >= 0 -> in window
        self.assertFalse(np.isnan(bull_dist[3]))
        # Bar 4: creation=0, i-lookback=1, so 0 < 1 -> out of window
        self.assertTrue(np.isnan(bull_dist[4]))


if __name__ == "__main__":
    unittest.main()
