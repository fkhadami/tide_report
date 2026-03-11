import unittest

import numpy as np
import pandas as pd

from analyze import (
    compute_fft_spectrum_hourly,
    require_complete_regular_series,
    run_utide_hourly,
)


def make_series(
    n: int,
    dt_hours: float = 1.0,
    missing_idx=None,
    drop_time_idx=None,
) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(hours=dt_hours))
    wl = np.sin(2.0 * np.pi * np.arange(n) / 12.0)
    df = pd.DataFrame({"time": time, "wl": wl})

    for idx in missing_idx or []:
        df.loc[idx, "wl"] = np.nan

    if drop_time_idx:
        df = df.drop(index=list(drop_time_idx)).reset_index(drop=True)

    return df


class AnalyzeValidationTests(unittest.TestCase):
    def test_require_complete_regular_series_accepts_clean_hourly_series(self):
        df = make_series(72)
        cleaned = require_complete_regular_series(df, expected_dt_hours=1.0)
        self.assertEqual(len(cleaned), 72)

    def test_require_complete_regular_series_rejects_missing_water_levels(self):
        df = make_series(72, missing_idx=[10])
        with self.assertRaisesRegex(ValueError, "Missing water levels detected"):
            require_complete_regular_series(df, expected_dt_hours=1.0)

    def test_require_complete_regular_series_rejects_irregular_timestamps(self):
        df = make_series(72, drop_time_idx=[10])
        with self.assertRaisesRegex(ValueError, "Irregular timestamps detected"):
            require_complete_regular_series(df, expected_dt_hours=1.0)

    def test_compute_fft_spectrum_runs_for_clean_series(self):
        df = make_series(96)
        fft = compute_fft_spectrum_hourly(df, period_min_h=2.0, period_max_h=24.0, dt_hours=1.0)
        self.assertGreater(len(fft["period_hours"]), 0)
        self.assertEqual(fft["n_points"], 96)

    def test_compute_fft_spectrum_rejects_gappy_series(self):
        df = make_series(96, missing_idx=[20])
        with self.assertRaisesRegex(ValueError, "Missing water levels detected"):
            compute_fft_spectrum_hourly(df, period_min_h=2.0, period_max_h=24.0, dt_hours=1.0)

    def test_run_utide_returns_reason_for_missing_water_levels(self):
        df = make_series(24 * 20, missing_idx=[20])
        result = run_utide_hourly(df, lat=-6.2, min_days=15.0, dt_hours=1.0)
        self.assertFalse(result["utide_ran"])
        self.assertIn("Missing water levels detected", result["reason"])

    def test_run_utide_returns_reason_for_irregular_timestamps(self):
        df = make_series(24 * 20, drop_time_idx=[20])
        result = run_utide_hourly(df, lat=-6.2, min_days=15.0, dt_hours=1.0)
        self.assertFalse(result["utide_ran"])
        self.assertIn("Irregular timestamps detected", result["reason"])


if __name__ == "__main__":
    unittest.main()
