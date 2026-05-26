from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ACTIVE_SPLIT_FILE, MANIFESTS_DIR, PROCESSED_DIR, SPLITS_DIR


ANALYTE_SPECS = {
    "MG": {
        "positive_col": "has_mg",
        "conc_col": "c_mg",
        "primary_peak": 1394,
        "support_peaks": [1170, 1617],
        "context_cols": ["c_thiram", "c_mba"],
    },
    "MBA": {
        "positive_col": "has_mba",
        "conc_col": "c_mba",
        "primary_peak": 1077,
        "support_peaks": [1180, 1587],
        "context_cols": ["c_thiram", "c_mg"],
    },
}


def intensity_at(spec: np.ndarray, wn: np.ndarray, center: float, half_width: int = 5) -> float:
    mask = (wn >= center - half_width) & (wn <= center + half_width)
    return float(spec[mask].mean())


def local_peak_score(spec: np.ndarray, wn: np.ndarray, center: float, peak_hw: int = 5, bg_hw: int = 15) -> float:
    peak = intensity_at(spec, wn, center, peak_hw)
    bg_left = intensity_at(spec, wn, center - bg_hw, 4)
    bg_right = intensity_at(spec, wn, center + bg_hw, 4)
    return float(peak - 0.5 * (bg_left + bg_right))


def normalize_bool_columns(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    for col in ["has_thiram", "has_mg", "has_mba", "is_design_sample", "is_blank", "is_background"]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
                .fillna(0)
                .astype(int)
            )
    return out


def compute_folder_qc(
    meta: pd.DataFrame,
    X_raw: np.ndarray,
    wn: np.ndarray,
    analyte_name: str,
    spec_cfg: dict,
) -> pd.DataFrame:
    positive_col = spec_cfg["positive_col"]
    conc_col = spec_cfg["conc_col"]
    primary_peak = spec_cfg["primary_peak"]
    support_peaks = spec_cfg["support_peaks"]

    positive_meta = meta.loc[meta[positive_col] == 1].copy()
    folder_rows = []

    for folder_name, grp in positive_meta.groupby("folder_name"):
        idx = grp.index.to_numpy()
        spectra = X_raw[idx]
        folder_mean = spectra.mean(axis=0)
        corr_to_mean = [np.corrcoef(spec, folder_mean)[0, 1] for spec in spectra]
        primary_scores = np.array([local_peak_score(spec, wn, primary_peak) for spec in spectra])
        support_scores = {
            peak: np.array([local_peak_score(spec, wn, peak) for spec in spectra]) for peak in support_peaks
        }
        composite_scores = np.column_stack([primary_scores] + [support_scores[peak] for peak in support_peaks]).mean(axis=1)
        first = grp.iloc[0]
        row = {
            "analyte": analyte_name,
            "folder_name": folder_name,
            "sample_count": int(len(grp)),
            "family": first.get("family", ""),
            "mixture_order": int(first.get("mixture_order", 0)),
            "mean_corr_to_folder_mean": float(np.mean(corr_to_mean)),
            "min_corr_to_folder_mean": float(np.min(corr_to_mean)),
            "primary_peak_cm1": primary_peak,
            "primary_peak_mean": float(primary_scores.mean()),
            "primary_peak_median": float(np.median(primary_scores)),
            "primary_peak_positive_frac": float((primary_scores > 0).mean()),
            "composite_peak_mean": float(composite_scores.mean()),
            "composite_peak_median": float(np.median(composite_scores)),
            "composite_peak_positive_frac": float((composite_scores > 0).mean()),
        }
        for col in [conc_col] + spec_cfg["context_cols"]:
            row[col] = int(first[col])
        for peak in support_peaks:
            scores = support_scores[peak]
            row[f"peak_{peak}_mean"] = float(scores.mean())
            row[f"peak_{peak}_positive_frac"] = float((scores > 0).mean())
        folder_rows.append(row)

    folder_df = pd.DataFrame(folder_rows)
    if folder_df.empty:
        return folder_df

    folder_df["flag_primary_negative_mean"] = folder_df["primary_peak_mean"] <= 0
    folder_df["flag_low_primary_positive_frac"] = folder_df["primary_peak_positive_frac"] < 0.5
    folder_df["flag_low_composite_positive_frac"] = folder_df["composite_peak_positive_frac"] < 0.5

    composite_q10 = folder_df["composite_peak_mean"].quantile(0.10)
    corr_q10 = folder_df["mean_corr_to_folder_mean"].quantile(0.10)
    folder_df["flag_low_composite_q10"] = folder_df["composite_peak_mean"] <= composite_q10
    folder_df["flag_low_corr_q10"] = folder_df["mean_corr_to_folder_mean"] <= corr_q10

    folder_df["review_flag_count"] = folder_df[
        [
            "flag_primary_negative_mean",
            "flag_low_primary_positive_frac",
            "flag_low_composite_positive_frac",
            "flag_low_composite_q10",
            "flag_low_corr_q10",
        ]
    ].sum(axis=1)
    folder_df["review_priority"] = np.select(
        [folder_df["review_flag_count"] >= 3, folder_df["review_flag_count"] >= 1],
        ["high", "medium"],
        default="low",
    )
    folder_df["review_reasons"] = folder_df.apply(build_reason_text, axis=1)

    return folder_df.sort_values(
        ["review_flag_count", "primary_peak_mean", "composite_peak_mean"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_reason_text(row: pd.Series) -> str:
    reasons = []
    if row["flag_primary_negative_mean"]:
        reasons.append("primary_peak_mean<=0")
    if row["flag_low_primary_positive_frac"]:
        reasons.append("primary_peak_positive_frac<0.5")
    if row["flag_low_composite_positive_frac"]:
        reasons.append("composite_peak_positive_frac<0.5")
    if row["flag_low_composite_q10"]:
        reasons.append("composite_peak_mean<=q10")
    if row["flag_low_corr_q10"]:
        reasons.append("mean_corr_to_folder_mean<=q10")
    return "; ".join(reasons) if reasons else "no_rule_triggered"


def compute_spectrum_review(meta: pd.DataFrame, X_raw: np.ndarray, folder_qc: pd.DataFrame) -> pd.DataFrame:
    flagged_folders = set(folder_qc.loc[folder_qc["review_priority"].isin(["high", "medium"]), "folder_name"])
    if not flagged_folders:
        return pd.DataFrame()

    rows = []
    for folder_name, grp in meta.loc[meta["folder_name"].isin(flagged_folders)].groupby("folder_name"):
        idx = grp.index.to_numpy()
        spectra = X_raw[idx]
        folder_mean = spectra.mean(axis=0)
        diff = np.abs(np.diff(spectra, axis=1))
        spike_score = diff.max(axis=1) / np.maximum(np.median(diff, axis=1), 1e-6)
        for local_i, (_, sample_row) in enumerate(grp.iterrows()):
            rows.append(
                {
                    "folder_name": folder_name,
                    "file_name": sample_row["file_name"],
                    "sample_id": sample_row["sample_id"],
                    "corr_to_folder_mean": float(np.corrcoef(spectra[local_i], folder_mean)[0, 1]),
                    "spike_score": float(spike_score[local_i]),
                }
            )
    sample_df = pd.DataFrame(rows)
    if sample_df.empty:
        return sample_df

    corr_q10 = sample_df["corr_to_folder_mean"].quantile(0.10)
    spike_q90 = sample_df["spike_score"].quantile(0.90)
    sample_df["flag_low_corr_q10"] = sample_df["corr_to_folder_mean"] <= corr_q10
    sample_df["flag_high_spike_q90"] = sample_df["spike_score"] >= spike_q90
    sample_df["review_flag_count"] = sample_df[["flag_low_corr_q10", "flag_high_spike_q90"]].sum(axis=1)
    sample_df["review_priority"] = np.select(
        [sample_df["review_flag_count"] >= 2, sample_df["review_flag_count"] >= 1],
        ["high", "medium"],
        default="low",
    )
    return sample_df.sort_values(["review_flag_count", "corr_to_folder_mean", "spike_score"], ascending=[False, True, False])


def write_markdown_summary(folder_frames: list[pd.DataFrame], output_path: Path) -> None:
    lines = [
        "# V7 QC Manifest Summary",
        "",
        "本清单只做规则化人工复核标记，不允许据此直接手改原始光谱。",
        "",
        "## 使用原则",
        "",
        "1. 先复核 high，再复核 medium。",
        "2. 复核动作只允许做保留、备注、排除，不允许修改原始 CSV。",
        "3. 若要排除 folder，必须把规则触发原因写回论文方法或补充材料。",
        "",
        "## 结果概览",
        "",
    ]

    for frame in folder_frames:
        analyte = frame.iloc[0]["analyte"] if not frame.empty else "Unknown"
        high_count = int((frame["review_priority"] == "high").sum()) if not frame.empty else 0
        medium_count = int((frame["review_priority"] == "medium").sum()) if not frame.empty else 0
        lines.extend(
            [
                f"### {analyte}",
                "",
                f"- positive folders: {len(frame)}",
                f"- high priority review: {high_count}",
                f"- medium priority review: {medium_count}",
                "",
            ]
        )
        if not frame.empty:
            top_rows = frame.loc[frame["review_priority"].isin(["high", "medium"]), ["folder_name", "review_priority", "review_reasons"]].head(8)
            if not top_rows.empty:
                lines.append("Top review folders:")
                lines.append("")
                for _, row in top_rows.iterrows():
                    lines.append(f"- {row['folder_name']} | {row['review_priority']} | {row['review_reasons']}")
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = normalize_bool_columns(pd.read_csv(SPLITS_DIR / ACTIVE_SPLIT_FILE))
    X_raw = np.load(PROCESSED_DIR / "X_raw.npy")
    wn = np.load(PROCESSED_DIR / "wavenumber.npy")

    folder_frames = []
    sample_frames = []
    for analyte_name, spec_cfg in ANALYTE_SPECS.items():
        folder_qc = compute_folder_qc(meta, X_raw, wn, analyte_name, spec_cfg)
        sample_qc = compute_spectrum_review(meta, X_raw, folder_qc)
        folder_frames.append(folder_qc)
        if not sample_qc.empty:
            sample_qc.insert(0, "analyte", analyte_name)
            sample_frames.append(sample_qc)

        folder_qc.to_csv(MANIFESTS_DIR / f"qc_folder_review_{analyte_name.lower()}.csv", index=False, encoding="utf-8-sig")
        if not sample_qc.empty:
            sample_qc.to_csv(MANIFESTS_DIR / f"qc_spectrum_review_{analyte_name.lower()}.csv", index=False, encoding="utf-8-sig")

    write_markdown_summary(folder_frames, MANIFESTS_DIR / "qc_manifest_summary.md")

    print(f"QC manifests written to: {MANIFESTS_DIR}")
    for frame in folder_frames:
        analyte = frame.iloc[0]["analyte"]
        high_count = int((frame["review_priority"] == "high").sum())
        medium_count = int((frame["review_priority"] == "medium").sum())
        print(f"{analyte}: folders={len(frame)}, high={high_count}, medium={medium_count}")


if __name__ == "__main__":
    main()