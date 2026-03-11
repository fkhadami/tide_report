# report_generator.py
from __future__ import annotations

import io
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors


def _draw_title(c: canvas.Canvas, text: str, x: float, y: float) -> float:
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, text)
    return y - 18


def _draw_section_header(c: canvas.Canvas, text: str, x: float, y: float) -> float:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, text)
    return y - 16


def _wrap_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    leading: float = 12,
    font_name: str = "Helvetica",
    font_size: int = 10,
) -> float:
    c.setFont(font_name, font_size)
    words = (text or "").split()
    line = ""
    for w in words:
        trial = (line + " " + w).strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            line = trial
        else:
            if line:
                c.drawString(x, y, line)
                y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def _draw_table(
    c: canvas.Canvas,
    x: float,
    y: float,
    col_widths: List[float],
    rows: List[List[str]],
    header_rows: int = 1,
    row_height: float = 16,
) -> float:
    if not rows:
        return y

    n_rows = len(rows)
    table_h = n_rows * row_height
    table_w = sum(col_widths)

    c.setStrokeColor(colors.black)
    c.rect(x, y - table_h, table_w, table_h, stroke=1, fill=0)

    for i in range(1, n_rows):
        yy = y - i * row_height
        c.line(x, yy, x + table_w, yy)

    xx = x
    for w in col_widths[:-1]:
        xx += w
        c.line(xx, y, xx, y - table_h)

    if header_rows > 0:
        c.setFillColor(colors.lightgrey)
        c.rect(x, y - header_rows * row_height, table_w, header_rows * row_height, stroke=0, fill=1)
        c.setFillColor(colors.black)

    for r_idx, row in enumerate(rows):
        yy = y - (r_idx + 0.75) * row_height
        is_header = r_idx < header_rows
        c.setFont("Helvetica-Bold" if is_header else "Helvetica", 10)
        xx = x + 4
        for c_idx, cell in enumerate(row):
            c.drawString(xx, yy, str(cell))
            xx += col_widths[c_idx]

    return y - table_h - 10


def _draw_image(
    c: canvas.Canvas,
    img_buf: io.BytesIO,
    x: float,
    y: float,
    width: float,
    height: float,
) -> float:
    try:
        img_buf.seek(0)
    except Exception:
        pass
    img = ImageReader(img_buf)
    c.drawImage(img, x, y - height, width=width, height=height, preserveAspectRatio=True, anchor="sw")
    return y - height - 12


def _new_page_with_header(c: canvas.Canvas, title: str, header: str, margin_x: float, top_y: float) -> float:
    c.showPage()
    y = top_y
    y = _draw_title(c, title, margin_x, y)
    y = _draw_section_header(c, header, margin_x, y)
    return y


def make_pdf_report(
    station_name: str,
    map_png: io.BytesIO,
    stats: Dict[str, Any],
    ts_raw_png: io.BytesIO,
    ts_clean_png: io.BytesIO,
    fft_png: io.BytesIO,
    fft_peaks: List[Dict[str, Any]],
    utide: Optional[Dict[str, Any]] = None,
    utide_fit_png: Optional[io.BytesIO] = None,
    narrative_text: Optional[str] = None,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    margin_x = 40
    top_y = page_h - 50
    bottom_y = 70
    content_w = page_w - 2 * margin_x

    title = f"Tidal report of {station_name}"

    # 1) MAP
    y = top_y
    y = _draw_title(c, title, margin_x, y)
    y = _draw_section_header(c, "1. Station Map", margin_x, y)
    y = _draw_image(c, map_png, margin_x, y, width=content_w, height=320)

    # 2) STATISTICS
    y = _new_page_with_header(c, title, "2. Statistics of the Data", margin_x, top_y)
    stats_rows = [
        ["Metric", "Value"],
        ["Start time", str(stats.get("start_time", "-"))],
        ["End time", str(stats.get("end_time", "-"))],
        ["Total rows", str(stats.get("n_total_rows", "-"))],
        ["Valid WL points", str(stats.get("n_valid_wl", "-"))],
        ["Missing rate (%)", f"{stats.get('missing_rate_percent', 0):.2f}"],
        ["Median dt (minutes)", f"{stats.get('dt_median_minutes', 0):.2f}"],
        ["Mean WL", f"{stats.get('mean', 0):.4f}"],
        ["Min WL", f"{stats.get('min', 0):.4f}"],
        ["Max WL", f"{stats.get('max', 0):.4f}"],
        ["Range (max-min)", f"{stats.get('range', 0):.4f}"],
        ["Spikes removed", str(stats.get("n_spikes_removed", "-"))],
    ]
    y = _draw_table(
        c,
        x=margin_x,
        y=y,
        col_widths=[content_w * 0.45, content_w * 0.55],
        rows=stats_rows,
        header_rows=1,
        row_height=16,
    )

    # 3) RAW TS
    y = _new_page_with_header(c, title, "3. Time Series of Raw Data", margin_x, top_y)
    y = _draw_image(c, ts_raw_png, margin_x, y, width=content_w, height=350)

    # 4) CLEAN TS
    y = _new_page_with_header(c, title, "4. Time Series after Spike Cleaning", margin_x, top_y)
    y = _draw_image(c, ts_clean_png, margin_x, y, width=content_w, height=350)

    # 5) FFT FIGURE + TABLE
    y = _new_page_with_header(c, title, "5. FFT Spectrum", margin_x, top_y)
    y = _draw_image(c, fft_png, margin_x, y, width=content_w, height=260)
    y = _draw_section_header(c, "6. FFT Peaks (Top 8)", margin_x, y)
    peak_rows = [["No", "Period (hours)", "Amplitude"]]
    for i, p in enumerate(fft_peaks[:8], start=1):
        peak_rows.append(
            [
                str(i),
                f"{float(p.get('period_hours', 0.0)):.3f}",
                f"{float(p.get('amplitude', 0.0)):.5f}",
            ]
        )
    y = _draw_table(
        c,
        x=margin_x,
        y=y,
        col_widths=[content_w * 0.15, content_w * 0.42, content_w * 0.43],
        rows=peak_rows,
        header_rows=1,
        row_height=16,
    )

    # 7) UTIDE FIGURE + TABLE
    y = _new_page_with_header(c, title, "7. Harmonic Analysis (UTide): Data vs Reconstruction", margin_x, top_y)
    if utide_fit_png is not None:
        y = _draw_image(c, utide_fit_png, margin_x, y, width=content_w, height=250)
    else:
        y = _wrap_text(c, "UTide reconstruction figure not available.", margin_x, y, max_width=content_w)

    y = _draw_section_header(c, "8. Dominant Constituents", margin_x, y)
    if utide and utide.get("utide_ran", False):
        cons = utide.get("constituents", []) or []
        cons_rows = [["Constituent", "Amplitude", "Phase (deg)"]]
        for row in cons:
            cons_rows.append(
                [
                    str(row.get("name", "")),
                    f"{float(row.get('amplitude', 0.0)):.4f}",
                    f"{float(row.get('phase_deg', 0.0)):.2f}",
                ]
            )
        y = _draw_table(
            c,
            x=margin_x,
            y=y,
            col_widths=[content_w * 0.34, content_w * 0.33, content_w * 0.33],
            rows=cons_rows,
            header_rows=1,
            row_height=16,
        )
    else:
        reason = (utide or {}).get("reason", "UTide result not available.")
        y = _wrap_text(c, f"UTide skipped/failed: {reason}", margin_x, y, max_width=content_w)

    # 9) LLM ANALYSIS
    if narrative_text:
        y = _new_page_with_header(c, title, "9. LLM Analysis", margin_x, top_y)
        for para in narrative_text.split("\n"):
            para = para.strip()
            if not para:
                y -= 8
            else:
                y = _wrap_text(c, para, margin_x, y, max_width=content_w, leading=12)
            if y < bottom_y:
                y = _new_page_with_header(c, title, "9. LLM Analysis (cont.)", margin_x, top_y)

    c.save()
    buf.seek(0)
    return buf.read()
