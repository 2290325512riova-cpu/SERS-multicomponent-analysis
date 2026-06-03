from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import originpro as op
import pandas as pd

from make_paper_figures import load_processed, mg_ratio_table, pure_mask
from paper_style import ANALYTE_COLORS, CONDITION_COLORS, ROOT, normalize


PAPER = ROOT / "data" / "pure63_mainline" / "models" / "paper_main"
OUT_DIR = ROOT / "figures" / "paper_main_origin"
ASCII_OUT = Path("D:/origin_codex_out/sers_ml_adjusted")


def add_line(gl, wks, colx: int, coly: int, color: str, width: float = 2.0):
    plot = gl.add_plot(wks, coly=coly, colx=colx)
    plot.color = color
    plot.set_float("line.width", width)
    return plot


def tidy_layer(gl, x_title: str, y_title: str, show_axis_titles: bool = True) -> None:
    try:
        gl.layer.activate()
    except Exception:
        try:
            gl.activate()
        except Exception:
            pass
    gl.axis("x").title = x_title if show_axis_titles else ""
    gl.axis("y").title = y_title if show_axis_titles else ""
    try:
        gl.remove_label("Legend")
    except Exception:
        pass
    for cmd in [
        'xb.font$="Arial"',
        'yl.font$="Arial"',
        "xb.fsize=9",
        "yl.fsize=9",
        "layer.x.label.font=font(arial)",
        "layer.y.label.font=font(arial)",
        "layer.x.label.size=6",
        "layer.y.label.size=6",
        "layer.x.label.bold=0",
        "layer.y.label.bold=0",
        "layer.x.thickness=1",
        "layer.y.thickness=1",
    ]:
        try:
            gl.lt_exec(cmd)
        except Exception:
            pass


def export_graph(gp, stem: str) -> None:
    ASCII_OUT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        gp.save_fig(str(ASCII_OUT / f"{stem}.{ext}"), type=ext, width=2400)
        shutil.copy2(ASCII_OUT / f"{stem}.{ext}", OUT_DIR / f"{stem}.{ext}")


def fig3() -> None:
    x, wn, meta = load_processed("p4")
    wks = op.new_sheet("w", lname="Fig3_pure_spectra_data")
    wks.from_list(0, wn.tolist(), lname="Raman shift", axis="X")
    for idx, analyte in enumerate(["Thiram", "MG", "4-MBA"], start=1):
        mask = pure_mask(meta, "MBA" if analyte == "4-MBA" else analyte, 4)
        y = normalize(x[mask].mean(axis=0)) + (3 - idx) * 1.25
        wks.from_list(idx, y.tolist(), lname=analyte, axis="Y")
    gp = op.new_graph(template="line", lname="Origin_Fig3_pure_spectra")
    gl = gp[0]
    add_line(gl, wks, 0, 1, ANALYTE_COLORS["Thiram"], 2.0)
    add_line(gl, wks, 0, 2, ANALYTE_COLORS["MG"], 2.0)
    add_line(gl, wks, 0, 3, ANALYTE_COLORS["MBA"], 2.0)
    gl.rescale()
    gl.set_xlim(400, 1800)
    gl.set_ylim(-0.10, 3.15)
    tidy_layer(gl, "Raman shift (cm^-1)", "Normalized intensity (offset)")
    export_graph(gp, "origin_clean_fig3_pure_spectra")


def fig5() -> None:
    x, wn, meta = load_processed("p4")
    masks = {
        "Pure MG": (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 0) & (meta.c_mg == 4),
        "MG+MBA": (meta.has_mg == 1) & (meta.has_thiram == 0) & (meta.has_mba == 1) & (meta.c_mg == 4),
        "MG+Thiram": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 0) & (meta.c_mg == 4),
        "Ternary": (meta.has_mg == 1) & (meta.has_thiram == 1) & (meta.has_mba == 1) & (meta.c_mg == 4),
    }
    means = {label: x[mask.values].mean(axis=0) for label, mask in masks.items()}
    base = means["Pure MG"]
    base_shift = float(np.percentile(base, 5))
    base_scale = float(np.max(base - base_shift)) or 1.0
    scaled = {label: (m - base_shift) / base_scale for label, m in means.items()}
    diff = scaled["Ternary"] - scaled["Pure MG"]
    band = (wn >= 1500) & (wn <= 1700)
    diff_scale = float(np.max(np.abs(diff[band]))) or 1.0
    _ = mg_ratio_table(x, wn, meta)

    wks = op.new_sheet("w", lname="Fig5_MG_suppression_data")
    wks.from_list(0, wn.tolist(), lname="Raman shift", axis="X")
    labels = ["Pure MG", "MG+MBA", "MG+Thiram", "Ternary"]
    for idx, label in enumerate(labels, start=1):
        wks.from_list(idx, scaled[label].tolist(), lname=label, axis="Y")
    wks.from_list(5, (diff / diff_scale * 0.16 - 0.28).tolist(), lname="Ternary - pure MG", axis="Y")

    gp = op.new_graph(template="line", lname="Origin_Fig5_MG_suppression")
    gl = gp[0]
    for idx, label in enumerate(labels, start=1):
        add_line(gl, wks, 0, idx, CONDITION_COLORS[label], 2.2 if label != "Ternary" else 2.8)
    add_line(gl, wks, 0, 5, ANALYTE_COLORS["MG"], 1.4)
    gl.rescale()
    gl.set_xlim(1500, 1700)
    gl.set_ylim(-0.42, 1.15)
    tidy_layer(gl, "Raman shift (cm^-1)", "Common-scaled intensity", show_axis_titles=False)
    export_graph(gp, "origin_clean_fig5_mg_suppression")


def fig8() -> None:
    soil_dir = PAPER / "soil_validation"
    x_soil = np.load(soil_dir / "X_soil_p1.npy")
    x_blank = np.load(soil_dir / "X_blanks_p1.npy")
    wn = np.load(soil_dir / "wavenumber.npy")
    meta = pd.read_csv(soil_dir / "soil_metadata.csv")
    ternary_mask = (meta.has_thiram == 1) & (meta.has_mg == 1) & (meta.has_mba == 1)
    soil_mean = normalize(x_soil[ternary_mask.values].mean(axis=0)) + 1.05
    blank_mean = normalize(x_blank.mean(axis=0))

    wks = op.new_sheet("w", lname="Fig8_soil_spectra_data")
    wks.from_list(0, wn.tolist(), lname="Raman shift", axis="X")
    wks.from_list(1, blank_mean.tolist(), lname="Soil blank", axis="Y")
    wks.from_list(2, soil_mean.tolist(), lname="Spiked ternary soil", axis="Y")
    gp = op.new_graph(template="line", lname="Origin_Fig8_soil_spectra")
    gl = gp[0]
    add_line(gl, wks, 0, 1, "#777777", 2.0)
    add_line(gl, wks, 0, 2, ANALYTE_COLORS["MG"], 2.2)
    gl.rescale()
    gl.set_xlim(400, 1800)
    gl.set_ylim(-0.05, 2.25)
    tidy_layer(gl, "Raman shift (cm^-1)", "Mean p1 spectra (offset)", show_axis_titles=False)
    export_graph(gp, "origin_clean_fig8_soil_spectra")


def main() -> None:
    ASCII_OUT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    op.set_show(True)
    op.new(asksave=False)
    fig3()
    fig5()
    fig8()
    op.save(str(ASCII_OUT / "sers_ml_origin_clean.opju"))
    shutil.copy2(ASCII_OUT / "sers_ml_origin_clean.opju", OUT_DIR / "sers_ml_origin_clean.opju")
    print("Wrote clean Origin project:")
    print(ASCII_OUT / "sers_ml_origin_clean.opju")
    op.detach()


if __name__ == "__main__":
    main()
