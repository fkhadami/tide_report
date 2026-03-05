# report_generator.py
# PDF report builder (updated):
# - Page 1: Basic statistics table + Time series figure
# - Page 2: FFT spectrum figure
# - Page 3: UTide constituents table + (optional) Observed vs UTide fit figure
# - Page 4+: LLM interpretation text (optional), auto page-break

from __future__ import annotations

import io
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors


# -------------------------
# Layout helpers
# -------------------------
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
    """
    Very simple text wrapper using stringWidth.
    Returns updated y after drawing.
    """
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
    """
    Draw a simple table (manual) and return updated y (below table).
    """
    if not rows:
        return y

    n_rows = len(rows)
    table_h = n_rows * row_height
    table_w = sum(col_widths)

    # Border
    c.setStrokeColor(colors.black)
    c.rect(x, y - table_h, table_w, table_h, stroke=1, fill=0)

    # Lines
    for i in range(1, n_rows):
        yy = y - i * row_height
        c.line(x, yy, x + table_w, yy)

    xx = x
    for w in col_widths[:-1]:
        xx += w
        c.line(xx, y, xx, y - table_h)

    # Header background
    if header_rows > 0:
        c.setFillColor(colors.lightgrey)
        c.rect(x, y - header_rows * row_height, table_w, header_rows * row_height, stroke=0, fill=1)
        c.setFillColor(colors.black)

    # Text
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
    """
    Draw image with bottom-left at (x, y-height). Return updated y below the image.
    """
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


# -------------------------
# Public API
# -------------------------
def make_pdf_report(
    report_title: str,
    stats: Dict[str, Any],
    ts_png: io.BytesIO,
    fft_png: io.BytesIO,
    utide: Optional[Dict[str, Any]] = None,
    utide_fit_png: Optional[io.BytesIO] = None,
    narrative_text: Optional[str] = None,
) -> bytes:
    """
    Build a multi-page PDF report.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4

    margin_x = 40
    top_y = page_h - 50
    bottom_y = 70
    content_w = page_w - 2 * margin_x

    # =========================
    # PAGE 1: Title + Stats table + Time series
    # =========================
    y = top_y
    y = _draw_title(c, report_title, margin_x, y)
    y = _draw_section_header(c, "1. Basic Statistics", margin_x, y)

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

    y = _draw_section_header(c, "2. Time Series", margin_x, y)
    img_h = 260
    y = _draw_image(c, ts_png, margin_x, y, width=content_w, height=img_h)

    # =========================
    # PAGE 2: FFT spectrum
    # =========================
    y = _new_page_with_header(c, report_title, "3. FFT Spectrum", margin_x, top_y)

    y = _wrap_text(
        c,
        "Amplitude spectrum ditampilkan sebagai fungsi periode (jam). "
        "Puncak dominan di sekitar ~12–13 jam sering terkait komponen semi-diurnal, "
        "sementara ~24–26 jam terkait komponen diurnal.",
        margin_x,
        y,
        max_width=content_w,
        leading=12,
    )

    img_h2 = 360
    y = _draw_image(c, fft_png, margin_x, y, width=content_w, height=img_h2)

    # =========================
    # PAGE 3: UTide constituents + optional fit plot
    # =========================
    y = _new_page_with_header(c, report_title, "4. Harmonic Analysis (UTide)", margin_x, top_y)

    if not utide:
        y = _wrap_text(c, "UTide results not available.", margin_x, y, max_width=content_w)
    else:
        ran = bool(utide.get("utide_ran", False))
        if not ran:
            reason = utide.get("reason", "UTide not run.")
            y = _wrap_text(c, f"UTide skipped: {reason}", margin_x, y, max_width=content_w)
        else:
            y = _wrap_text(
                c,
                f"UTide ran successfully. Duration: {utide.get('duration_days', 0):.2f} days. "
                f"Latitude: {utide.get('lat', 'None')}. Points used: {utide.get('n_points', '-')}.",
                margin_x,
                y,
                max_width=content_w,
            )

            constituents = utide.get("constituents", []) or []
            table_rows = [["Constituent", "Amplitude", "Phase (deg)"]]
            for row in constituents:
                table_rows.append(
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
                rows=table_rows,
                header_rows=1,
                row_height=16,
            )

            if utide_fit_png is not None and y > (bottom_y + 120):
                y = _draw_section_header(c, "UTide Reconstruction (Observed vs Fit)", margin_x, y)
                img_h3 = min(260, max(160, y - bottom_y))
                y = _draw_image(c, utide_fit_png, margin_x, y, width=content_w, height=img_h3)

    # =========================
    # PAGE 4+: LLM Interpretation (optional)
    # =========================
    if narrative_text:
        y = _new_page_with_header(c, report_title, "5. Interpretation (LLM)", margin_x, top_y)

        # print paragraph by paragraph; auto page-break
        for para in narrative_text.split("\n"):
            para = para.strip()
            if not para:
                y -= 8
            else:
                y = _wrap_text(c, para, margin_x, y, max_width=content_w, leading=12)
            if y < bottom_y:
                y = _new_page_with_header(c, report_title, "5. Interpretation (LLM) (cont.)", margin_x, top_y)

    c.save()
    buf.seek(0)
    return buf.read()
