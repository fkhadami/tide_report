# app.py
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analyze import (
    load_and_clean,
    require_complete_regular_series,
    summarize_stats,
    clean_spikes_hampel,
    plot_station_map_png,
    plot_timeseries_png,
    plot_fft_spectrum_png_hourly,
    compute_fft_spectrum_hourly,
    find_fft_peaks,
    run_utide_hourly,
    plot_utide_fit_png,
)
from report_generator import make_pdf_report
from llm_writer import generate_narrative


st.set_page_config(page_title="Tide Agent", layout="centered")
st.title("Tide Agent — Pasang Surut Report Generator")

st.write("Alur kerja: 1) Analyze & Preview, 2) User approve, 3) Generate PDF.")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
USER_MANUAL_PATH = Path("Tide Agent User Manual.pdf")
SAMPLE_CSV = """time,wl
2024-01-01 00:00:00,1.12
2024-01-01 01:00:00,1.34
2024-01-01 02:00:00,1.08
2024-01-01 03:00:00,0.95
2024-01-01 04:00:00,0.89
2024-01-01 05:00:00,1.01
"""


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _buf(data: bytes) -> io.BytesIO:
    b = io.BytesIO(data)
    b.seek(0)
    return b


def _check_item(ok: bool, label: str, detail: str) -> None:
    status = "OK" if ok else "Needs attention"
    st.markdown(f"- **{label}**: {status}. {detail}")


def build_upload_checks(
    file_bytes: bytes,
    time_col: str,
    wl_col: str,
    dt_hours: float,
    lat: float | None,
    lon: float | None,
    run_utide: bool,
    min_days: float,
) -> tuple[pd.DataFrame | None, list[tuple[bool, str, str]], str | None]:
    checks: list[tuple[bool, str, str]] = []

    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return None, [(False, "CSV can be read", f"File could not be parsed as CSV: {e}")], None

    has_time = time_col in raw.columns
    has_wl = wl_col in raw.columns
    checks.append((has_time and has_wl, "Required columns", f"Expected `{time_col}` and `{wl_col}`. Found: {list(raw.columns)}"))
    if not (has_time and has_wl):
        return None, checks, None

    df = load_and_clean(io.BytesIO(file_bytes), time_col=time_col, wl_col=wl_col)
    checks.append((len(df) > 0, "Valid timestamp rows", f"{len(df)} rows remain after parsing timestamps."))

    invalid_time_count = int(pd.to_datetime(raw[time_col], errors="coerce").isna().sum())
    checks.append((invalid_time_count == 0, "Timestamp format", f"{invalid_time_count} rows could not be parsed as time."))

    wl_numeric = pd.to_numeric(raw[wl_col], errors="coerce")
    non_numeric_wl_count = int((raw[wl_col].notna() & wl_numeric.isna()).sum())
    checks.append((non_numeric_wl_count == 0, "Water level values", f"{non_numeric_wl_count} rows are non-numeric."))

    duplicate_count = int(df["time"].duplicated(keep=False).sum())
    checks.append((duplicate_count == 0, "Duplicate timestamps", f"{duplicate_count} duplicate timestamp rows detected."))

    missing_wl_count = int(df["wl"].isna().sum())
    checks.append((missing_wl_count == 0, "Missing water levels", f"{missing_wl_count} rows have missing water levels."))

    regular_series_error = None
    try:
        require_complete_regular_series(df, expected_dt_hours=float(dt_hours))
        checks.append((True, "Regular sampling interval", f"Matches configured `dt={dt_hours:.4f}` hours."))
    except Exception as e:
        regular_series_error = str(e)
        checks.append((False, "Regular sampling interval", regular_series_error))

    has_coordinates = lat is not None and lon is not None
    checks.append((has_coordinates, "Station coordinates", "Latitude and longitude are required for the station map."))

    if run_utide:
        duration_days = 0.0
        dfv = df.dropna(subset=["wl"]).sort_values("time")
        if len(dfv) >= 2:
            duration_days = (dfv["time"].iloc[-1] - dfv["time"].iloc[0]).total_seconds() / 86400.0
        utide_ready = has_coordinates and regular_series_error is None and duration_days >= float(min_days)
        checks.append((utide_ready, "UTide readiness", f"Duration is {duration_days:.2f} days; minimum configured is {float(min_days):.2f} days."))

    return df, checks, regular_series_error


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")

station_name = st.sidebar.text_input("Station name", value="Station A")
time_col = st.sidebar.text_input("Nama kolom waktu", value="time")
wl_col = st.sidebar.text_input("Nama kolom water level", value="wl")

lat_str = st.sidebar.text_input("Latitude (deg)", value="")
lon_str = st.sidebar.text_input("Longitude (deg)", value="")
dt_hours = st.sidebar.number_input("Sampling dt (hours)", min_value=0.01, max_value=24.0, value=1.0, step=0.01)

lat = None
if lat_str.strip():
    try:
        lat = float(lat_str.strip())
    except ValueError:
        st.sidebar.error("Latitude harus angka.")

lon = None
if lon_str.strip():
    try:
        lon = float(lon_str.strip())
    except ValueError:
        st.sidebar.error("Longitude harus angka.")

st.sidebar.divider()
st.sidebar.subheader("Spike Cleaning")
hampel_window = st.sidebar.number_input("Hampel window (odd)", min_value=3, max_value=101, value=9, step=2)
hampel_sigma = st.sidebar.number_input("Hampel n-sigma", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

st.sidebar.divider()
st.sidebar.subheader("FFT")
period_min_h = st.sidebar.number_input("Min period (hours)", min_value=0.5, max_value=200.0, value=2.0, step=0.5)
period_max_h = st.sidebar.number_input("Max period (hours)", min_value=1.0, max_value=500.0, value=60.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("UTide")
run_utide = st.sidebar.checkbox("Run UTide", value=True)
min_days = st.sidebar.number_input("Min duration (days) for UTide", min_value=1.0, max_value=60.0, value=15.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("LLM (ChatGPT via OpenAI API)")
use_llm = st.sidebar.checkbox("Generate interpretation with LLM", value=False)
llm_model = st.sidebar.text_input("Model", value="gpt-5-nano")
llm_api_key = ""
if use_llm:
    llm_api_key = st.sidebar.text_input(
        "OpenAI API Key (optional)",
        value="",
        type="password",
        placeholder="sk-...",
        help="Jika kosong, app akan mencoba OPENAI_API_KEY dari environment.",
    )


# -------------------------
# Main
# -------------------------
with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
Prepare the following before upload:
- CSV file with a time column and a water-level column
- Station name
- Latitude and longitude for the station map
- Correct sampling interval in hours

Data rules:
- Default column names are `time` and `wl`
- Time values should look like `2024-01-01 00:00:00`
- Water-level values must be numeric
- FFT and UTide need one sample per timestamp, no missing water levels, and regular sampling
- UTide works best when the data duration is long enough for the configured minimum days

Outputs:
- Summary statistics
- Raw and cleaned time series plots
- FFT spectrum and peak periods
- UTide reconstruction and constituents
- PDF report
"""
    )
    c1, c2 = st.columns(2)
    c1.download_button(
        label="Download sample CSV",
        data=SAMPLE_CSV.encode("utf-8"),
        file_name="sample_tide_data.csv",
        mime="text/csv",
    )
    if USER_MANUAL_PATH.exists():
        c2.download_button(
            label="Download user manual",
            data=USER_MANUAL_PATH.read_bytes(),
            file_name=USER_MANUAL_PATH.name,
            mime="application/pdf",
        )

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    with st.expander("Pre-upload checklist", expanded=True):
        _check_item(bool(station_name.strip()), "Station name", "Fill in the station name in the sidebar.")
        _check_item(lat is not None and lon is not None, "Coordinates", "Enter latitude and longitude to generate the station map.")
        _check_item(float(dt_hours) > 0, "Sampling interval", "Set the expected interval between samples in hours.")
        _check_item(True, "CSV format", f"Prepare columns named `{time_col}` and `{wl_col}`.")
    st.info("Silakan upload CSV untuk mulai.")
    st.stop()

file_bytes = uploaded.getvalue()

df_preview, upload_checks, regular_series_error = build_upload_checks(
    file_bytes=file_bytes,
    time_col=time_col,
    wl_col=wl_col,
    dt_hours=float(dt_hours),
    lat=lat,
    lon=lon,
    run_utide=run_utide,
    min_days=float(min_days),
)

with st.expander("Pre-upload checklist", expanded=True):
    for ok, label, detail in upload_checks:
        _check_item(ok, label, detail)
    if regular_series_error:
        st.caption("FFT and UTide may be skipped until the sampling interval or missing-data issue is fixed.")

with st.expander("Preview data (10 baris pertama)"):
    try:
        if df_preview is None:
            raise ValueError("Preview is not available because the uploaded file does not pass the basic column checks.")
        st.dataframe(df_preview.head(10), width="stretch")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

st.divider()

if st.button("Analyze & Preview Report"):
    try:
        if not station_name.strip():
            raise ValueError("Station name wajib diisi.")
        if lat is None or lon is None:
            raise ValueError("Latitude dan Longitude wajib diisi untuk membuat peta stasiun.")

        df_raw = load_and_clean(io.BytesIO(file_bytes), time_col=time_col, wl_col=wl_col)

        spike_res = clean_spikes_hampel(
            df_raw,
            window_size=int(hampel_window),
            n_sigma=float(hampel_sigma),
        )
        df_clean = spike_res["df_clean"]

        stats = summarize_stats(df_clean)
        stats["station_name"] = station_name.strip()
        stats["n_spikes_removed"] = int(spike_res["n_spikes_removed"])

        map_png = plot_station_map_png(station_name.strip(), lat=float(lat), lon=float(lon), target_scale_km=10.0)
        ts_raw_png = plot_timeseries_png(df_raw, title="Raw Time Series")
        ts_clean_png = plot_timeseries_png(df_clean, title="Time Series after Spike Cleaning")

        fft_png = plot_fft_spectrum_png_hourly(
            df_clean,
            period_min_h=float(period_min_h),
            period_max_h=float(period_max_h),
            dt_hours=float(dt_hours),
        )
        fft_res = compute_fft_spectrum_hourly(
            df_clean,
            period_min_h=float(period_min_h),
            period_max_h=float(period_max_h),
            dt_hours=float(dt_hours),
        )
        fft_peaks = find_fft_peaks(fft_res, top_k=8, min_separation_hours=0.8)

        utide_out = None
        utide_fit_png = None
        if run_utide:
            ut_raw = run_utide_hourly(
                df_clean,
                lat=float(lat),
                min_days=float(min_days),
                constituents_top_k=12,
                dt_hours=float(dt_hours),
            )
            utide_out = {k: v for k, v in ut_raw.items() if not str(k).startswith("_")}
            if ut_raw.get("utide_ran", False):
                utide_fit_png = plot_utide_fit_png(df_clean, ut_raw)

        narrative = None
        if use_llm:
            try:
                narrative = generate_narrative(
                    stats=stats,
                    fft_peaks=fft_peaks,
                    utide=utide_out,
                    model=llm_model.strip() or "gpt-5.2",
                    api_key=llm_api_key.strip() or None,
                )
            except Exception as e:
                narrative = f"(LLM narrative failed: {e})"

        st.session_state["preview_data"] = {
            "station_name": station_name.strip(),
            "lat": float(lat),
            "lon": float(lon),
            "dt_hours": float(dt_hours),
            "stats": stats,
            "fft_peaks": fft_peaks,
            "utide": utide_out,
            "narrative": narrative,
            "map_png_bytes": map_png.getvalue(),
            "ts_raw_png_bytes": ts_raw_png.getvalue(),
            "ts_clean_png_bytes": ts_clean_png.getvalue(),
            "fft_png_bytes": fft_png.getvalue(),
            "utide_fit_png_bytes": utide_fit_png.getvalue() if utide_fit_png is not None else None,
        }
        st.success("Preview report berhasil dibuat. Silakan review hasil di bawah.")
    except Exception as e:
        st.error(f"Gagal saat analisis: {e}")


if "preview_data" in st.session_state:
    p = st.session_state["preview_data"]

    st.divider()
    st.subheader(f"Tidal report of {p['station_name']}")
    st.caption(f"Lat: {p['lat']:.6f}, Lon: {p['lon']:.6f} | dt: {p['dt_hours']:.4f} hours")

    st.markdown("### 1) Map")
    st.image(p["map_png_bytes"], width="stretch")

    st.markdown("### 2) Statistics of the Data")
    stats = p["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid WL", f"{stats.get('n_valid_wl', '-')}")
    c2.metric("Missing (%)", f"{stats.get('missing_rate_percent', 0.0):.2f}")
    c3.metric("Median dt (min)", f"{stats.get('dt_median_minutes', 0.0):.2f}")
    c4.metric("Spikes removed", f"{stats.get('n_spikes_removed', 0)}")
    c5, c6, c7 = st.columns(3)
    c5.metric("Mean", f"{stats.get('mean', 0.0):.4f}")
    c6.metric("Min", f"{stats.get('min', 0.0):.4f}")
    c7.metric("Max", f"{stats.get('max', 0.0):.4f}")
    st.metric("Range", f"{stats.get('range', 0.0):.4f}")
    st.caption(f"Start: {stats.get('start_time', '-')}")
    st.caption(f"End: {stats.get('end_time', '-')}")

    st.markdown("### 3) Figure Time Series of Raw Data")
    st.image(p["ts_raw_png_bytes"], width="stretch")

    st.markdown("### 4) Figure Time Series after Cleaning Spike")
    st.image(p["ts_clean_png_bytes"], width="stretch")

    st.markdown("### 5) Figure FFT Spectrum")
    st.image(p["fft_png_bytes"], width="stretch")

    st.markdown("### 6) Analysis of FFT Result (8 Peaks)")
    if p["fft_peaks"]:
        st.dataframe(p["fft_peaks"], width="stretch")
    else:
        st.info("Tidak ada puncak FFT terdeteksi.")

    st.markdown("### 7) Harmonics Analysis Result (UTide: Data vs Reconstruction)")
    utide_out = p["utide"]
    if utide_out is None:
        st.info("UTide tidak dijalankan.")
    elif not utide_out.get("utide_ran", False):
        st.info(f"UTide skipped: {utide_out.get('reason', '-')}")
    else:
        if p.get("utide_fit_png_bytes"):
            st.image(p["utide_fit_png_bytes"], width="stretch")

    st.markdown("### 8) Table of Dominant Constituents")
    if utide_out and utide_out.get("utide_ran", False):
        st.dataframe(utide_out.get("constituents", []), width="stretch")
    else:
        st.info("Tabel konstituen tidak tersedia.")

    st.markdown("### 9) LLM Result Analysis")
    if p["narrative"]:
        st.write(p["narrative"])
    else:
        st.info("LLM analysis tidak dijalankan.")

    st.divider()
    user_accept = st.checkbox("Saya sudah review preview report dan siap generate PDF.")
    if st.button("Generate PDF & Download", disabled=not user_accept):
        try:
            run_id = new_run_id()
            run_dir = OUTPUTS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "stats.json").write_text(json.dumps(p["stats"], indent=2), encoding="utf-8")
            (run_dir / "fft_peaks.json").write_text(json.dumps(p["fft_peaks"], indent=2), encoding="utf-8")
            if p["utide"] is not None:
                (run_dir / "utide.json").write_text(json.dumps(p["utide"], indent=2), encoding="utf-8")
            if p["narrative"]:
                (run_dir / "narrative.txt").write_text(p["narrative"], encoding="utf-8")

            (run_dir / "map.png").write_bytes(p["map_png_bytes"])
            (run_dir / "timeseries_raw.png").write_bytes(p["ts_raw_png_bytes"])
            (run_dir / "timeseries_clean.png").write_bytes(p["ts_clean_png_bytes"])
            (run_dir / "fft_spectrum.png").write_bytes(p["fft_png_bytes"])
            if p.get("utide_fit_png_bytes"):
                (run_dir / "utide_fit.png").write_bytes(p["utide_fit_png_bytes"])

            pdf_bytes = make_pdf_report(
                station_name=p["station_name"],
                map_png=_buf(p["map_png_bytes"]),
                stats=p["stats"],
                ts_raw_png=_buf(p["ts_raw_png_bytes"]),
                ts_clean_png=_buf(p["ts_clean_png_bytes"]),
                fft_png=_buf(p["fft_png_bytes"]),
                fft_peaks=p["fft_peaks"],
                utide=p["utide"],
                utide_fit_png=_buf(p["utide_fit_png_bytes"]) if p.get("utide_fit_png_bytes") else None,
                narrative_text=p["narrative"],
            )

            pdf_path = run_dir / "tide_report.pdf"
            pdf_path.write_bytes(pdf_bytes)

            st.success("PDF report berhasil dibuat.")
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"tide_report_{run_id}.pdf",
                mime="application/pdf",
            )
            with st.expander("Output files saved"):
                st.write(str(run_dir))
                st.write("Files:", sorted([f.name for f in run_dir.iterdir()]))
        except Exception as e:
            st.error(f"Gagal generate PDF: {e}")
