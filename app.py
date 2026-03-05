# app.py
# Tide Agent (Streamlit):
# - Upload CSV (time, wl)
# - Stats + Time Series PNG
# - FFT Spectrum PNG (configurable sampling dt)
# - UTide harmonic analysis (configurable sampling dt, optional) + optional fit plot
# - Optional LLM narrative (ChatGPT via OpenAI API) using summary + FFT peaks + UTide constituents
# - PDF report with tables, figures, and LLM interpretation
# - Saves artifacts to outputs/<run_id>/

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from analyze import (
    load_and_clean,
    summarize_stats,
    plot_timeseries_png,
    plot_fft_spectrum_png_hourly,
    compute_fft_spectrum_hourly,
    find_fft_peaks,
    run_utide_hourly,
    plot_utide_fit_png,
)
from report_generator import make_pdf_report

# LLM optional
from llm_writer import generate_narrative


# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Tide Agent", layout="centered")
st.title("Tide Agent — Pasang Surut Report Generator")

st.write(
    "Upload CSV berisi kolom waktu dan tinggi muka air. "
    "Atur interval sampling data (`dt`) di sidebar sesuai data Anda."
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")

time_col = st.sidebar.text_input("Nama kolom waktu", value="time")
wl_col = st.sidebar.text_input("Nama kolom water level", value="wl")
dt_hours = st.sidebar.number_input("Sampling dt (hours)", min_value=0.01, max_value=24.0, value=1.0, step=0.01)

st.sidebar.divider()
st.sidebar.subheader("FFT")

period_min_h = st.sidebar.number_input("Min period (hours)", min_value=0.5, max_value=200.0, value=2.0, step=0.5)
period_max_h = st.sidebar.number_input("Max period (hours)", min_value=1.0, max_value=500.0, value=60.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("UTide")

run_utide = st.sidebar.checkbox("Run UTide", value=True)
lat_str = st.sidebar.text_input("Latitude (deg) — optional", value="")
min_days = st.sidebar.number_input("Min duration (days) for UTide", min_value=1.0, max_value=60.0, value=15.0, step=1.0)

lat = None
if lat_str.strip():
    try:
        lat = float(lat_str.strip())
    except ValueError:
        st.sidebar.error("Latitude harus angka (mis. -6.9). Kosongkan jika tidak dipakai.")

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
        help="Jika dikosongkan, app akan mencoba OPENAI_API_KEY dari environment.",
    )
st.sidebar.caption("Masukkan API key di atas atau set OPENAI_API_KEY di environment.")


# -------------------------
# Main UI
# -------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Silakan upload CSV untuk mulai.")
    st.stop()

file_bytes = uploaded.getvalue()

with st.expander("Preview data (10 baris pertama)"):
    try:
        df_preview = load_and_clean(io.BytesIO(file_bytes), time_col=time_col, wl_col=wl_col)
        st.dataframe(df_preview.head(10))
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

st.divider()

if st.button("Generate Report"):
    run_id = new_run_id()
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Load & stats
        df = load_and_clean(io.BytesIO(file_bytes), time_col=time_col, wl_col=wl_col)
        stats = summarize_stats(df)
        (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        # 2) Figures: time series + FFT plot
        ts_png = plot_timeseries_png(df)
        fft_png = plot_fft_spectrum_png_hourly(
            df,
            period_min_h=float(period_min_h),
            period_max_h=float(period_max_h),
            dt_hours=float(dt_hours),
        )

        (run_dir / "timeseries.png").write_bytes(ts_png.getvalue())
        (run_dir / "fft_spectrum.png").write_bytes(fft_png.getvalue())

        # 3) FFT peaks (for LLM interpretation)
        fft_res = compute_fft_spectrum_hourly(
            df,
            period_min_h=float(period_min_h),
            period_max_h=float(period_max_h),
            dt_hours=float(dt_hours),
        )
        fft_peaks = find_fft_peaks(fft_res, top_k=6, min_separation_hours=0.8)
        (run_dir / "fft_peaks.json").write_text(json.dumps(fft_peaks, indent=2), encoding="utf-8")

        # 4) UTide optional
        utide_out = None
        utide_fit_png = None

        if run_utide:
            ut_raw = run_utide_hourly(
                df,
                lat=lat,
                min_days=float(min_days),
                constituents_top_k=12,
                dt_hours=float(dt_hours),
            )
            utide_out = {k: v for k, v in ut_raw.items() if not str(k).startswith("_")}
            (run_dir / "utide.json").write_text(json.dumps(utide_out, indent=2), encoding="utf-8")

            if ut_raw.get("utide_ran", False):
                utide_fit_png = plot_utide_fit_png(df, ut_raw)
                (run_dir / "utide_fit.png").write_bytes(utide_fit_png.getvalue())

        # 5) LLM narrative optional
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
                # Do not fail report if LLM fails
                narrative = f"(LLM narrative failed: {e})"

            (run_dir / "narrative.txt").write_text(narrative or "", encoding="utf-8")

        # 6) Build PDF
        pdf_bytes = make_pdf_report(
            report_title="Tide Data Processing Report",
            stats=stats,
            ts_png=ts_png,
            fft_png=fft_png,
            utide=utide_out,
            utide_fit_png=utide_fit_png,
            narrative_text=narrative,
        )

        pdf_path = run_dir / "tide_report.pdf"
        pdf_path.write_bytes(pdf_bytes)

        st.success("Report berhasil dibuat.")
        st.session_state["report_data"] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stats": stats,
            "dt_hours": float(dt_hours),
            "fft_peaks": fft_peaks,
            "utide": utide_out,
            "narrative": narrative,
            "timeseries_png_bytes": ts_png.getvalue(),
            "fft_png_bytes": fft_png.getvalue(),
            "utide_fit_png_bytes": utide_fit_png.getvalue() if utide_fit_png is not None else None,
            "pdf_bytes": pdf_bytes,
        }
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"tide_report_{run_id}.pdf",
            mime="application/pdf",
        )

        with st.expander("Output files saved"):
            st.write(str(run_dir))
            st.write("Files:", sorted([p.name for p in run_dir.iterdir()]))

    except Exception as e:
        st.error(f"Gagal: {e}")


if "report_data" in st.session_state:
    report = st.session_state["report_data"]
    stats = report["stats"]
    fft_peaks = report["fft_peaks"]
    utide_out = report["utide"]
    narrative = report["narrative"]

    st.divider()
    st.subheader("Report di Dashboard")
    st.caption(f"Run ID: {report.get('run_id', '-')}")
    st.caption(f"Sampling dt: {report.get('dt_hours', 1.0):.4f} hours")

    st.download_button(
        label="Download PDF",
        data=report["pdf_bytes"],
        file_name=f"tide_report_{report.get('run_id', 'latest')}.pdf",
        mime="application/pdf",
        key="download_pdf_bottom",
    )

    st.markdown("### 1) Basic Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid WL Points", f"{stats.get('n_valid_wl', '-')}")
    c2.metric("Missing Rate (%)", f"{stats.get('missing_rate_percent', 0.0):.2f}")
    c3.metric("Median dt (min)", f"{stats.get('dt_median_minutes', 0.0):.2f}")

    st.caption(f"Start: {stats.get('start_time', '-')}")
    st.caption(f"End: {stats.get('end_time', '-')}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Mean WL", f"{stats.get('mean', 0.0):.4f}")
    c5.metric("Min WL", f"{stats.get('min', 0.0):.4f}")
    c6.metric("Max WL", f"{stats.get('max', 0.0):.4f}")

    st.metric("Range (max-min)", f"{stats.get('range', 0.0):.4f}")

    st.markdown("### 2) Time Series")
    st.image(report["timeseries_png_bytes"], use_container_width=True)

    st.markdown("### 3) FFT Spectrum")
    st.image(report["fft_png_bytes"], use_container_width=True)
    st.markdown("#### Dominant FFT Peaks (hours)")
    if fft_peaks:
        st.dataframe(fft_peaks, use_container_width=True)
    else:
        st.info("Tidak ada puncak FFT yang terdeteksi.")

    st.markdown("### 4) Harmonic Analysis (UTide)")
    if not utide_out:
        st.info("UTide tidak dijalankan.")
    elif not utide_out.get("utide_ran", False):
        st.info(f"UTide skipped: {utide_out.get('reason', '-')}")
    else:
        st.write(
            f"Duration: {utide_out.get('duration_days', 0.0):.2f} days | "
            f"Latitude: {utide_out.get('lat', 'None')} | "
            f"Points: {utide_out.get('n_points', '-')}"
        )
        st.dataframe(utide_out.get("constituents", []), use_container_width=True)
        if report.get("utide_fit_png_bytes"):
            st.image(report["utide_fit_png_bytes"], use_container_width=True)

    if narrative:
        st.markdown("### 5) Interpretation (LLM)")
        st.write(narrative)
