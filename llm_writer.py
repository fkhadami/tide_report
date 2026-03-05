# llm_writer.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

from openai import OpenAI

def build_prompt(
    stats: Dict[str, Any],
    fft_peaks: Optional[List[Dict[str, float]]],
    utide: Optional[Dict[str, Any]],
) -> str:
    peaks_txt = "- (no peaks provided)"
    if fft_peaks:
        peaks_txt = "\n".join([f"- {p['period_hours']:.2f} h (amp {p['amplitude']:.4f})" for p in fft_peaks])

    utide_txt = "UTide: not provided."
    if utide is not None:
        if not utide.get("utide_ran", False):
            utide_txt = f"UTide: skipped. Reason: {utide.get('reason','-')}"
        else:
            cons = (utide.get("constituents") or [])[:12]
            utide_txt = "UTide constituents (top):\n" + "\n".join(
                [f"- {c['name']}: A={c['amplitude']:.4f}, phase={c['phase_deg']:.2f} deg" for c in cons]
            )

    return f"""
Anda adalah asisten ilmiah oseanografi fisika. Tulis interpretasi hasil pengolahan data pasang surut.
Gunakan bahasa Indonesia ilmiah, ringkas, dan hanya berdasarkan angka berikut.

STATISTIK:
- Start: {stats.get('start_time')}
- End: {stats.get('end_time')}
- N valid: {stats.get('n_valid_wl')}
- Missing (%): {stats.get('missing_rate_percent'):.2f}
- Median dt (min): {stats.get('dt_median_minutes'):.2f}
- Mean: {stats.get('mean'):.4f}
- Min: {stats.get('min'):.4f}
- Max: {stats.get('max'):.4f}
- Range: {stats.get('range'):.4f}

FFT PEAKS (period hours):
{peaks_txt}

{utide_txt}

Struktur output:
1) Ringkasan data & kualitas (1 paragraf).
2) Deskripsi time series (1 paragraf).
3) Interpretasi FFT (1 paragraf; semi-diurnal/diurnal/mixed).
4) Jika UTide ada: interpretasi konstituen dominan (1 paragraf).
5) Keterbatasan & rekomendasi (bullet 3–5 poin).

Larangan: jangan menambah angka baru yang tidak ada; jika tidak cukup info, tulis “tidak dapat disimpulkan”.
""".strip()


def generate_narrative(
    stats: Dict[str, Any],
    fft_peaks: Optional[List[Dict[str, float]]] = None,
    utide: Optional[Dict[str, Any]] = None,
    model: str = "gpt-5-nano",
    api_key: Optional[str] = None,
) -> str:
    prompt = build_prompt(stats, fft_peaks, utide)
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text
