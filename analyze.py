# analyze.py
# Tide analysis module: loading/cleaning, basic statistics, FFT, and UTide harmonic analysis.
# Expected dataframe columns after loading: time (datetime64), wl (float)

from __future__ import annotations

import io
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

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


def clean_spikes_hampel(
    df: pd.DataFrame,
    window_size: int = 9,
    n_sigma: float = 3.0,
) -> Dict[str, Any]:
    """
    Hampel spike filter:
    - detect outliers using rolling median + MAD
    - replace detected spikes with rolling median
    Returns:
      {
        "df_clean": DataFrame(time, wl),
        "n_spikes_removed": int
      }
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3.")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be > 0.")

    dfv = df.copy()
    s = dfv["wl"].astype(float)
    med = s.rolling(window=window_size, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window=window_size, center=True, min_periods=1).median()
    sigma = 1.4826 * mad

    spike_mask = (s - med).abs() > (n_sigma * sigma)
    spike_mask = spike_mask.fillna(False)

    s_clean = s.copy()
    s_clean[spike_mask] = med[spike_mask]
    dfv["wl"] = s_clean

    return {
        "df_clean": dfv,
        "n_spikes_removed": int(spike_mask.sum()),
    }


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


def require_complete_regular_series(
    df: pd.DataFrame,
    expected_dt_hours: float,
    tolerance_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Return a sorted copy suitable for FFT/UTide.

    Spectral and harmonic methods require a complete regularly sampled series.
    This helper rejects:
    - duplicate timestamps
    - missing water levels
    - timestamp gaps/irregular spacing
    """
    if expected_dt_hours <= 0:
        raise ValueError("expected_dt_hours must be > 0.")

    dfv = df.sort_values("time").reset_index(drop=True).copy()
    if len(dfv) < 2:
        raise ValueError("Not enough data to evaluate sampling interval.")

    duplicated = dfv["time"].duplicated(keep=False)
    if duplicated.any():
        n_dup = int(duplicated.sum())
        raise ValueError(
            f"Duplicate timestamps detected ({n_dup} rows). "
            "FFT/UTide require one sample per timestamp."
        )

    missing_mask = dfv["wl"].isna()
    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        raise ValueError(
            f"Missing water levels detected ({n_missing} rows). "
            "FFT/UTide require a complete regular series; fill or resample gaps first."
        )

    dt = dfv["time"].diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        raise ValueError("Cannot determine sampling interval.")

    expected_dt_seconds = float(expected_dt_hours) * 3600.0
    irregular = (dt - expected_dt_seconds).abs() > tolerance_seconds
    if irregular.any():
        dt_unique_hours = sorted({round(v / 3600.0, 6) for v in dt[irregular].tolist()[:5]})
        raise ValueError(
            "Irregular timestamps detected. "
            f"Expected dt={expected_dt_hours:.4f} hours, found non-matching intervals such as {dt_unique_hours} hours."
        )

    return dfv


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
    dfv = require_complete_regular_series(df, expected_dt_hours=dt_hours)
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
    try:
        dfv = require_complete_regular_series(df, expected_dt_hours=dt_hours)
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

    try:
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
    except Exception as e:
        return {
            "utide_ran": False,
            "reason": f"UTide failed: {e}",
            "duration_days": float(duration_days),
            "constituents": [],
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
def plot_timeseries_png(df: pd.DataFrame, title: str = "Tide time series") -> io.BytesIO:
    dfv = df.dropna(subset=["wl"]).sort_values("time").copy()
    buf = io.BytesIO()
    plt.figure()
    plt.plot(dfv["time"], dfv["wl"])
    plt.xlabel("Time")
    plt.ylabel("Water level")
    plt.title(title)
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


def plot_station_map_png(
    station_name: str,
    lat: float,
    lon: float,
    target_scale_km: float = 10.0,
) -> io.BytesIO:
    """
    Draw a simple local station map with:
    - lat/lon grid
    - station marker
    - dynamic scale bar in km
    """
    lat = float(lat)
    lon = float(lon)
    target_scale_km = max(1.0, float(target_scale_km))

    deg_lat_per_km = 1.0 / 111.32
    cos_lat = max(0.1, float(np.cos(np.deg2rad(lat))))
    deg_lon_per_km = 1.0 / (111.32 * cos_lat)

    def _bounds(half_width_km: float, half_height_km: float):
        dlat = half_height_km * deg_lat_per_km
        dlon = half_width_km * deg_lon_per_km
        return lon - dlon, lon + dlon, lat - dlat, lat + dlat

    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy  # type: ignore
        from cartopy.feature import ShapelyFeature  # type: ignore
        from cartopy.io import shapereader  # type: ignore
        from shapely.geometry import box  # type: ignore

        # Prefer a project-local cache so datasets can be bundled offline,
        # while still allowing Cartopy's downloader when data is missing.
        project_data_dir = os.getenv(
            "TIDE_AGENT_CARTOPY_DATA_DIR",
            str((Path(__file__).resolve().parent / "data" / "cartopy")),
        )
        if os.path.isdir(project_data_dir):
            cartopy.config["data_dir"] = project_data_dir
            cartopy.config["pre_existing_data_dir"] = project_data_dir

        def _bounded_feature(
            bbox,
            category: str,
            name: str,
            resolution: str,
            facecolor: str,
            edgecolor: str,
            linewidth: float = 0.5,
        ):
            shp_path = shapereader.natural_earth(
                resolution=resolution,
                category=category,
                name=name,
            )
            reader = shapereader.Reader(shp_path)
            geoms = []
            for geom in reader.geometries():
                if geom is None or geom.is_empty or not geom.intersects(bbox):
                    continue
                clipped = geom.intersection(bbox)
                if not clipped.is_empty:
                    geoms.append(clipped)
            if not geoms:
                return None
            return ShapelyFeature(
                geoms,
                ccrs.PlateCarree(),
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )

        half_width_km = 50.0
        half_height_km = 50.0
        land_feature = None
        coastline_feature = None
        map_extent_label = "100 km x 100 km"

        for search_half_width_km in [50.0, 100.0, 200.0, 400.0]:
            search_half_height_km = search_half_width_km
            lon_min, lon_max, lat_min, lat_max = _bounds(search_half_width_km, search_half_height_km)
            bbox = box(lon_min, lat_min, lon_max, lat_max)
            land_feature = _bounded_feature(
                bbox=bbox,
                category="physical",
                name="land",
                resolution="10m",
                facecolor="#b0b0b0",
                edgecolor="none",
            )
            coastline_feature = _bounded_feature(
                bbox=bbox,
                category="physical",
                name="coastline",
                resolution="10m",
                facecolor="none",
                edgecolor="black",
                linewidth=0.6,
            )
            if land_feature is not None or coastline_feature is not None:
                half_width_km = search_half_width_km
                half_height_km = search_half_height_km
                map_extent_label = f"{int(half_width_km * 2)} km x {int(half_height_km * 2)} km"
                break

        lon_min, lon_max, lat_min, lat_max = _bounds(half_width_km, half_height_km)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_facecolor("white")

        if land_feature is not None:
            ax.add_feature(land_feature, zorder=2)
        if coastline_feature is not None:
            ax.add_feature(coastline_feature, zorder=3)

        ax.scatter([lon], [lat], color="red", s=60, zorder=5, transform=ccrs.PlateCarree())
        ax.text(lon, lat, f"  {station_name}", va="bottom", fontsize=9, transform=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f"Station Location Map (Natural Earth 10m, {map_extent_label}): {station_name}")

        # Scale bar
        scale_km = min(target_scale_km, half_width_km * 0.7)
        scale_deg_lon = scale_km * deg_lon_per_km
        x0 = lon_min + (lon_max - lon_min) * 0.06
        y0 = lat_min + (lat_max - lat_min) * 0.08
        x1 = x0 + scale_deg_lon
        ax.plot([x0, x1], [y0, y0], color="black", linewidth=3, transform=ccrs.PlateCarree(), zorder=6)
        ax.text(
            (x0 + x1) / 2.0,
            y0 + (lat_max - lat_min) * 0.03,
            f"{scale_km:.0f} km",
            ha="center",
            fontsize=9,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        # Fallback to simple land mask if GSHHS/cartopy not available.
        try:
            from global_land_mask import globe  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Map rendering needs cartopy (GSHHS) or global-land-mask fallback."
            ) from e

        n_grid = 300
        lons = np.linspace(lon_min, lon_max, n_grid)
        lats = np.linspace(lat_min, lat_max, n_grid)
        lon2d, lat2d = np.meshgrid(lons, lats)
        land_mask = globe.is_land(lat2d, lon2d).astype(float)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(
            lon2d,
            lat2d,
            land_mask,
            levels=[-0.5, 0.5, 1.5],
            colors=["white", "#b0b0b0"],
            alpha=1.0,
        )
        ax.contour(lon2d, lat2d, land_mask, levels=[0.5], colors="black", linewidths=0.6, alpha=0.6)
        ax.scatter([lon], [lat], color="red", s=60, zorder=5)
        ax.text(lon, lat, f"  {station_name}", va="bottom", fontsize=9)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"Station Location Map (fallback, 100 km x 100 km): {station_name}")

        scale_km = min(target_scale_km, half_width_km * 0.7)
        scale_deg_lon = scale_km * deg_lon_per_km
        x0 = lon_min + (lon_max - lon_min) * 0.06
        y0 = lat_min + (lat_max - lat_min) * 0.08
        x1 = x0 + scale_deg_lon
        ax.plot([x0, x1], [y0, y0], color="black", linewidth=3)
        ax.text((x0 + x1) / 2.0, y0 + (lat_max - lat_min) * 0.03, f"{scale_km:.0f} km", ha="center", fontsize=9)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        buf.seek(0)
        return buf


def _extract_rtide_prediction(pred_raw: Any, expected_len: int) -> np.ndarray:
    """
    Coerce RTide Predict(...) output to a 1D float array of size expected_len.
    """
    if isinstance(pred_raw, pd.Series):
        arr = pred_raw.to_numpy(dtype=float)
    elif isinstance(pred_raw, pd.DataFrame):
        preferred = ["prediction", "predictions", "pred", "yhat", "fit", "fitted"]
        col = next((c for c in preferred if c in pred_raw.columns), None)
        if col is None:
            num_cols = pred_raw.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                raise ValueError("RTide prediction dataframe has no numeric columns.")
            col = num_cols[0]
        arr = pred_raw[col].to_numpy(dtype=float)
    elif isinstance(pred_raw, dict):
        for k in ["prediction", "predictions", "pred", "yhat", "fit", "fitted"]:
            if k in pred_raw:
                arr = np.asarray(pred_raw[k], dtype=float)
                break
        else:
            raise ValueError("RTide prediction dict does not contain recognized prediction keys.")
    else:
        arr = np.asarray(pred_raw, dtype=float)

    arr = np.ravel(arr)
    if arr.size != expected_len:
        raise ValueError(
            f"RTide prediction length mismatch. Expected {expected_len}, got {arr.size}."
        )
    return arr


def run_rtide_analysis(
    df: pd.DataFrame,
    lat: Optional[float],
    lon: Optional[float],
    min_days: float = 15.0,
    dt_hours: float = 1.0,
    max_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run RTide (if installed) and return a serializable result + internal arrays.
    """
    dfv = df.dropna(subset=["wl"]).sort_values("time").copy()
    try:
        assert_regular_sampling(dfv, expected_dt_hours=dt_hours)
    except ValueError as e:
        return {
            "rtide_ran": False,
            "reason": str(e),
            "duration_days": 0.0,
        }

    if lat is None or lon is None:
        return {
            "rtide_ran": False,
            "reason": "RTide requires latitude and longitude.",
            "duration_days": 0.0,
        }

    if len(dfv) < 48:
        return {
            "rtide_ran": False,
            "reason": "Too few valid points for RTide.",
            "duration_days": 0.0,
        }

    duration_days = (dfv["time"].iloc[-1] - dfv["time"].iloc[0]).total_seconds() / 86400.0
    if duration_days < min_days:
        return {
            "rtide_ran": False,
            "reason": f"Duration {duration_days:.2f} days < {min_days:.2f} days (skip RTide).",
            "duration_days": float(duration_days),
        }

    n_original = int(len(dfv))
    if max_points is not None and max_points > 0 and len(dfv) > max_points:
        idx = np.linspace(0, len(dfv) - 1, num=int(max_points), dtype=int)
        dfv = dfv.iloc[idx].copy()

    try:
        from rtide import RTide  # type: ignore
    except ImportError:
        return {
            "rtide_ran": False,
            "reason": "Package 'rtide' is not installed. Install with: pip install rtide",
            "duration_days": float(duration_days),
        }

    try:
        rt_df = pd.DataFrame(
            {"observations": dfv["wl"].to_numpy(dtype=float)},
            index=pd.to_datetime(dfv["time"]),
        )

        model = RTide(rt_df, float(lat), float(lon))
        model.Prepare_Inputs()
        model.Train()

        pred_raw = model.Predict(rt_df.copy())
        pred = _extract_rtide_prediction(pred_raw, expected_len=len(rt_df))

        obs = rt_df["observations"].to_numpy(dtype=float)
        residual = obs - pred
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        mae = float(np.mean(np.abs(residual)))

        return {
            "rtide_ran": True,
            "duration_days": float(duration_days),
            "n_original_points": n_original,
            "n_points": int(len(rt_df)),
            "lat": float(lat),
            "lon": float(lon),
            "rmse": rmse,
            "mae": mae,
            "_time": pd.to_datetime(dfv["time"]).to_numpy(),
            "_obs": obs,
            "_pred": pred,
            "_model": model,
        }
    except Exception as e:
        return {
            "rtide_ran": False,
            "reason": f"RTide failed: {e}",
            "duration_days": float(duration_days),
        }


def plot_rtide_fit_png(rtide_result: Dict[str, Any]) -> io.BytesIO:
    if not rtide_result.get("rtide_ran", False):
        raise ValueError("RTide not run.")

    t = pd.to_datetime(rtide_result["_time"])
    obs = np.asarray(rtide_result["_obs"], dtype=float)
    pred = np.asarray(rtide_result["_pred"], dtype=float)

    buf = io.BytesIO()
    plt.figure()
    plt.plot(t, obs, label="Observed")
    plt.plot(t, pred, label="RTide fit")
    plt.xlabel("Time")
    plt.ylabel("Water level")
    plt.title("Observed vs RTide Prediction")
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
