# analyze.py
# Tide analysis module: loading/cleaning, basic statistics, FFT, and UTide harmonic analysis.
# Expected dataframe columns after loading: time (datetime64), wl (float)

from __future__ import annotations

import io
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utide import solve, reconstruct


# -------------------------
# 1) Load & clean
# -------------------------
def load_and_clean(
    uploaded_file,
    time_col: str = "time",
    wl_col: str = "wl",
) -> pd.DataFrame:
    """
    Load CSV and standardize to columns: time, wl
    - time parsed to datetime
    - wl parsed to float
    - rows with invalid time removed
    - sorted by time
    """
    df = pd.read_csv(uploaded_file)

    if time_col not in df.columns or wl_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{time_col}' and '{wl_col}'. Found: {list(df.columns)}"
        )

    df = df[[time_col, wl_col]].rename(columns={time_col: "time", wl_col: "wl"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["wl"] = pd.to_numeric(df["wl"], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# -------------------------
# 2) Basic statistics + sampling checks
# -------------------------
def _median_dt_seconds(time_series: pd.Series) -> float:
    dt = time_series.diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        return float("nan")
    return float(np.median(dt))


def summarize_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns a deterministic stats dict.
    Uses non-missing wl for value stats; missing-rate computed over all rows with valid time.
    """
    if len(df) == 0:
        raise ValueError("No rows found after cleaning.")

    missing_rate = float(df["wl"].isna().mean() * 100.0)

    dfv = df.dropna(subset=["wl"]).copy()
    if len(dfv) < 2:
        raise ValueError("Not enough valid water level data after cleaning (need >=2).")

    dt_med_sec = _median_dt_seconds(dfv["time"])
    dt_minutes = dt_med_sec / 60.0 if np.isfinite(dt_med_sec) else float("nan")

    stats = {
        "start_time": str(dfv["time"].iloc[0]),
        "end_time": str(dfv["time"].iloc[-1]),
        "n_total_rows": int(len(df)),
        "n_valid_wl": int(len(dfv)),
        "missing_rate_percent": missing_rate,
        "dt_median_minutes": float(dt_minutes),
        "mean": float(dfv["wl"].mean()),
        "min": float(dfv["wl"].min()),
        "max": float(dfv["wl"].max()),
        "range": float(dfv["wl"].max() - dfv["wl"].min()),
    }
    return stats


def assert_regular_sampling(
    df: pd.DataFrame,
    expected_dt_hours: float,
    tolerance_seconds: float = 60.0,
) -> None:
    """
    Raise ValueError if median dt is not close to expected_dt_hours within tolerance.
    """
    if expected_dt_hours <= 0:
        raise ValueError("expected_dt_hours must be > 0.")

    dfv = df.dropna(subset=["wl"]).sort_values("time")
    if len(dfv) < 2:
        raise ValueError("Not enough valid data to check sampling interval.")
    dt_med = _median_dt_seconds(dfv["time"])
    if not np.isfinite(dt_med):
        raise ValueError("Cannot determine sampling interval (dt is NaN).")
    expected_dt_seconds = float(expected_dt_hours) * 3600.0
    if abs(dt_med - expected_dt_seconds) > tolerance_seconds:
        raise ValueError(
            f"Sampling interval mismatch. Expected dt={expected_dt_hours:.4f} hours, "
            f"median dt={dt_med/3600.0:.4f} hours."
        )


def assert_hourly(df: pd.DataFrame, tolerance_seconds: float = 60.0) -> None:
    """
    Backward-compatible wrapper for hourly checks.
    """
    assert_regular_sampling(df, expected_dt_hours=1.0, tolerance_seconds=tolerance_seconds)


# -------------------------
# 3) FFT spectrum
# -------------------------
def compute_fft_spectrum_hourly(
    df: pd.DataFrame,
    period_min_h: float = 2.0,
    period_max_h: float = 60.0,
    detrend: str = "mean",
    window: str = "hann",
    dt_hours: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute one-sided FFT amplitude spectrum for regularly sampled data.
    Returns dict with arrays serialized as python lists:
      - period_hours
      - amplitude
    """
    assert_regular_sampling(df, expected_dt_hours=dt_hours)

    dfv = df.dropna(subset=["wl"]).sort_values("time").copy()
    x = dfv["wl"].to_numpy(dtype=float)

    if len(x) < 48:
        raise ValueError("Data too short for meaningful FFT (recommend >= 48 points).")

    # detrend
    if detrend == "mean":
        x = x - np.nanmean(x)
    elif detrend == "none":
        pass
    else:
        raise ValueError("detrend must be 'mean' or 'none'.")

    n = len(x)

    # window
    if window == "hann":
        w = np.hanning(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError("window must be 'hann' or 'none'.")

    X = np.fft.rfft(x * w)
    freq_cph = np.fft.rfftfreq(n, d=float(dt_hours))  # cycles per hour

    # amplitude scaling (practical normalization)
    amp = (2.0 / np.sum(w)) * np.abs(X)

    # drop zero frequency
    freq_cph = freq_cph[1:]
    amp = amp[1:]

    period_h = 1.0 / freq_cph

    # filter period range
    m = (period_h >= period_min_h) & (period_h <= period_max_h)
    period_h = period_h[m]
    amp = amp[m]

    # sort by period ascending
    idx = np.argsort(period_h)
    period_h = period_h[idx]
    amp = amp[idx]

    return {
        "period_hours": period_h.tolist(),
        "amplitude": amp.tolist(),
        "period_min_h": float(period_min_h),
        "period_max_h": float(period_max_h),
        "n_points": int(n),
        "sampling_dt_hours": float(dt_hours),
        "window": window,
        "detrend": detrend,
    }


def find_fft_peaks(
    fft_result: Dict[str, Any],
    top_k: int = 6,
    min_separation_hours: float = 0.8,
) -> List[Dict[str, float]]:
    """
    Simple peak picker without scipy:
    - local maxima
    - then keep top_k by amplitude
    - enforce minimum separation in period space
    """
    period = np.array(fft_result["period_hours"], dtype=float)
    amp = np.array(fft_result["amplitude"], dtype=float)

    if len(period) < 3:
        return []

    # local maxima
    is_peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] > amp[2:])
    peak_idx = np.where(is_peak)[0] + 1
    if len(peak_idx) == 0:
        return []

    # sort candidates by amplitude desc
    cand = sorted(peak_idx, key=lambda i: amp[i], reverse=True)

    selected: List[int] = []
    for i in cand:
        if len(selected) >= top_k:
            break
        if all(abs(period[i] - period[j]) >= min_separation_hours for j in selected):
            selected.append(i)

    peaks = [{"period_hours": float(period[i]), "amplitude": float(amp[i])} for i in selected]
    # sort by period ascending for readability
    peaks = sorted(peaks, key=lambda d: d["period_hours"])
    return peaks


# -------------------------
# 4) UTide harmonic analysis
# -------------------------


def run_utide_hourly(
    df: pd.DataFrame,
    lat: Optional[float] = None,
    min_days: float = 15.0,
    constituents_top_k: int = 12,
    dt_hours: float = 1.0,
) -> Dict[str, Any]:
    dfv = df.dropna(subset=["wl"]).sort_values("time").copy()
    try:
        assert_regular_sampling(dfv, expected_dt_hours=dt_hours)
    except ValueError as e:
        return {
            "utide_ran": False,
            "reason": str(e),
            "duration_days": 0.0,
            "constituents": [],
        }

    if len(dfv) < 48:
        return {"utide_ran": False, "reason": "Too few valid points for UTide.", "duration_days": 0.0, "constituents": []}

    duration_days = (dfv["time"].iloc[-1] - dfv["time"].iloc[0]).total_seconds() / 86400.0
    if duration_days < min_days:
        return {
            "utide_ran": False,
            "reason": f"Duration {duration_days:.2f} days < {min_days:.2f} days (skip UTide).",
            "duration_days": float(duration_days),
            "constituents": [],
        }

    # ✅ gunakan datetime64 langsung dan simpan untuk reconstruct
    t = pd.to_datetime(dfv["time"]).to_numpy()          # datetime64[ns]
    u = dfv["wl"].to_numpy(dtype=float)

    coef = solve(
        t, u,
        lat=lat,
        method="ols",
        conf_int="linear",
        trend=False,
        Rayleigh_min=1.0,
    )

    names = [str(n) for n in coef.name]
    amps = np.array(coef.A, dtype=float)
    phases = np.array(coef.g, dtype=float)

    order = np.argsort(amps)[::-1]
    top = order[:constituents_top_k] if len(order) > constituents_top_k else order

    constituents = [{
        "name": names[i],
        "amplitude": float(amps[i]),
        "phase_deg": float(phases[i]),
    } for i in top]

    return {
        "utide_ran": True,
        "duration_days": float(duration_days),
        "t0": str(dfv["time"].iloc[0]),
        "n_points": int(len(dfv)),
        "lat": None if lat is None else float(lat),
        "constituents": constituents,

        # internal (kunci agar reconstruct tidak mismatch)
        "_coef": coef,
        "_t": t,
        "_u": u,
    }


def utide_reconstruct_series(utide_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Return DataFrame with:
      - time
      - tide_fit
      - residual (observed - fit)
    """
    if not utide_result.get("utide_ran", False):
        raise ValueError("UTide was not run; cannot reconstruct.")

    coef = utide_result["_coef"]
    time = pd.to_datetime(utide_result["_t"])
    u = np.array(utide_result["_u"], dtype=float)

    rec = reconstruct(time.to_numpy(), coef)
    fit = np.array(rec.h, dtype=float)

    return pd.DataFrame(
        {
            "time": time,
            "tide_fit": fit,
            "residual": u - fit,
        }
    )


# -------------------------
# 5) Plot helpers (PNG bytes)
# -------------------------
def plot_timeseries_png(df: pd.DataFrame) -> io.BytesIO:
    dfv = df.dropna(subset=["wl"]).sort_values("time").copy()
    buf = io.BytesIO()
    plt.figure()
    plt.plot(dfv["time"], dfv["wl"])
    plt.xlabel("Time")
    plt.ylabel("Water level")
    plt.title("Tide time series")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return buf


def plot_fft_spectrum_png_hourly(
    df: pd.DataFrame,
    period_min_h: float = 2.0,
    period_max_h: float = 60.0,
    dt_hours: float = 1.0,
) -> io.BytesIO:
    fft_res = compute_fft_spectrum_hourly(
        df,
        period_min_h=period_min_h,
        period_max_h=period_max_h,
        dt_hours=dt_hours,
    )
    period = np.array(fft_res["period_hours"], dtype=float)
    amp = np.array(fft_res["amplitude"], dtype=float)

    buf = io.BytesIO()
    plt.figure()
    plt.plot(period, amp)
    plt.xlabel("Period (hours)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(f"FFT Spectrum (dt={dt_hours:g} h)")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return buf


def plot_utide_fit_png(df: pd.DataFrame, utide_result: Dict[str, Any]) -> io.BytesIO:
    if not utide_result.get("utide_ran", False):
        raise ValueError("UTide not run.")

    coef = utide_result["_coef"]
    t = utide_result["_t"]     # ✅ persis time yang dipakai solve()
    u = utide_result["_u"]

    rec = reconstruct(t, coef)
    fit = np.array(rec.h, dtype=float)

    # sanity check: kalau fit hampir konstan, fail-fast
    if np.nanstd(fit) < 1e-6:
        raise ValueError("UTide reconstruction is nearly constant (time mismatch likely).")

    buf = io.BytesIO()
    plt.figure()
    plt.plot(t, u, label="Observed")
    plt.plot(t, fit, label="UTide fit")
    plt.xlabel("Time")
    plt.ylabel("Water level")
    plt.title("Observed vs UTide Reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return buf



# -------------------------
# 6) One-call convenience (optional)
# -------------------------
def analyze_all_hourly(
    uploaded_file,
    time_col: str = "time",
    wl_col: str = "wl",
    lat: Optional[float] = None,
    run_utide: bool = True,
    dt_hours: float = 1.0,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
      - load & clean
      - stats
      - FFT (+ peaks)
      - UTide (optional)
    Returns a dict suitable to be saved as JSON (internal UTide objects removed).
    """
    df = load_and_clean(uploaded_file, time_col=time_col, wl_col=wl_col)
    stats = summarize_stats(df)

    fft_res = compute_fft_spectrum_hourly(df, period_min_h=2.0, period_max_h=60.0, dt_hours=dt_hours)
    fft_peaks = find_fft_peaks(fft_res, top_k=6, min_separation_hours=0.8)

    out: Dict[str, Any] = {
        "stats": stats,
        "fft": {**fft_res, "peaks": fft_peaks},
    }

    if run_utide:
        ut = run_utide_hourly(df, lat=lat, min_days=15.0, constituents_top_k=12, dt_hours=dt_hours)
        # strip internal keys for JSON
        out["utide"] = {k: v for k, v in ut.items() if not str(k).startswith("_")}

    return out
