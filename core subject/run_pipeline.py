"""
run_pipeline.py — Main entry point for the SERS multi-component analysis pipeline.
Strategy 3: MBA treated as a regular component (equal to Thiram and MG).

Usage:
    python run_pipeline.py                     # Full pipeline
    python run_pipeline.py --step data         # Only data preparation
    python run_pipeline.py --step eda          # Only EDA plots
    python run_pipeline.py --step train        # Only training & evaluation
    python run_pipeline.py --step report       # Only generate report
    python run_pipeline.py --rebuild           # Force rebuild (ignore caches)
    python run_pipeline.py --models RF SVM     # Only run specific models
    python run_pipeline.py --preprocess p1     # Use a registered preprocessing variant
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (
    TASKS, TASKS_SUPPLEMENTARY, TASKS_MT_FULL,
    MODELS_DIR, REPORTS_DIR, N_FOLDS, SEED,
    PROCESSED_DIR, SPLITS_DIR, ACTIVE_SPLIT_FILE, DATA_VERSION, TASK_PROFILE,
    PREPROCESS_TAGS,
)
from src.dataset import build_metadata, load_and_preprocess, create_cv_splits, load_preprocessed_variants
from src.models import MODEL_REGISTRY
from src.train_eval import run_cv_evaluation, aggregate_results, get_best_models, save_results, load_predictions
from src.visualize import (
    plot_eda, plot_preprocessing_comparison, plot_confusion_matrices,
    plot_model_comparison, plot_split_distribution,
)


def resolve_result_dir(result_scope: str) -> Path:
    """Map a logical result scope to the target models directory."""
    return MODELS_DIR / result_scope


def step_data(args):
    """Step 1: Build metadata, preprocess spectra, create CV splits."""
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)

    print("\n[1.1] Building metadata from folder structure...")
    meta = build_metadata()
    print(f"  Total spectra: {len(meta)}")
    print(f"  Folders: {meta['folder_name'].nunique()}")
    print(f"  Families: {meta['family'].value_counts().to_dict()}")

    print("\n[1.2] Loading and preprocessing spectra...")
    wn, X_dict = load_and_preprocess(meta, rebuild=args.rebuild, return_dict=True)
    first_X = next(iter(X_dict.values()))
    print(f"  Shape: {first_X.shape} ({first_X.shape[1]} wavenumber points)")

    print("\n[1.3] Creating CV splits...")
    meta = create_cv_splits(meta, rebuild=args.rebuild, split_filename=args.split_file)

    return meta, wn, X_dict


def step_eda(meta, wn, X_dict):
    """Step 2: Exploratory Data Analysis plots."""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    plot_eda(meta, wn, X_dict['raw'], X_dict['p1'])
    plot_preprocessing_comparison(wn, X_dict)
    plot_split_distribution(meta)


def step_train(meta, X_dict, args):
    """Step 3: Model training and evaluation."""
    print("\n" + "=" * 60)
    print("STEP 3: Model Training & Evaluation")
    print("=" * 60)

    preprocess_tag = args.preprocess
    result_dir = resolve_result_dir(args.result_scope)
    X = X_dict[preprocess_tag]
    model_names = args.models if args.models else list(MODEL_REGISTRY.keys())
    task_group = args.task_group
    if task_group == 'main':
        active_tasks = TASKS
    elif task_group == 'supplementary':
        active_tasks = TASKS_SUPPLEMENTARY
    else:
        active_tasks = TASKS_MT_FULL

    print(f"\n  Preprocessing: {preprocess_tag}")
    print(f"  Models: {model_names}")
    print(f"  Task group: {task_group}")
    print(f"  Tasks: {len(active_tasks)} tasks")
    print(f"  CV: {N_FOLDS}-fold StratifiedGroupKFold")
    print(f"  Result scope: {args.result_scope}")
    print(f"  Result dir: {result_dir}")

    res_df = run_cv_evaluation(
        meta,
        X,
        model_names=model_names,
        preprocess_tag=preprocess_tag,
        tasks_override=active_tasks,
    )
    agg_df = aggregate_results(res_df)
    best_df = get_best_models(agg_df)

    save_results(res_df, agg_df, tag=preprocess_tag, output_dir=result_dir)

    print("\n  === Summary (Best Model per Task) ===")
    for _, row in best_df.iterrows():
        print(f"  {row['TaskName']:25s} → {row['Model']:10s} "
              f"F1={row['F1_mean']:.3f}±{row['F1_std']:.3f}")

    return res_df, agg_df


def step_report(meta, agg_df, predictions, result_scope='quick_check'):
    """Step 4: Generate figures and markdown report."""
    print("\n" + "=" * 60)
    print("STEP 4: Visualization & Report")
    print("=" * 60)

    plot_confusion_matrices(agg_df, predictions, meta)
    plot_model_comparison(agg_df)
    generate_report(meta, agg_df, result_scope=result_scope)


def generate_report(meta, agg_df, result_scope='quick_check'):
    """Generate a markdown evaluation report (Chinese)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_scope_suffix = '' if result_scope == 'quick_check' else f'_{result_scope}'

    best_df = get_best_models(agg_df)
    n_spectra = len(meta)
    n_folders = meta['folder_name'].nunique()
    task_lookup = {task['id']: task for task in TASKS_MT_FULL}
    ordered_tasks = [task for task in TASKS_MT_FULL if task['id'] in set(agg_df['Task'])]

    # Task name mapping to Chinese
    task_cn = {
        'Thiram Presence': '福美双有无',
        'MG Presence': '孔雀石绿有无',
        'MBA Presence': 'MBA有无',
        'Thiram Positive Molar Grade': '福美双阳性摩尔分级',
        'MG Positive Molar Grade': '孔雀石绿阳性摩尔分级',
        'MBA Positive Molar Grade': 'MBA阳性摩尔分级',
        'Mixture Complexity': '混合复杂度',
    }

    # Family name mapping
    family_cn = {
        'single': '单组分',
        'binary_Thiram_MG': '二元(福美双+孔雀石绿)',
        'binary_Thiram_MBA': '二元(福美双+MBA)',
        'binary_MG_MBA': '二元(孔雀石绿+MBA)',
        'ternary': '三元',
    }

    report = f"""# SERS 多组分分类评估报告
> 生成时间: {ts}
> 策略: **MBA 作为常规组分** (策略三)
> 交叉验证: {N_FOLDS}折 StratifiedGroupKFold (group=文件夹, stratify=混合类型)

## 1. 数据集概览
| 指标 | 值 |
|------|-----|
| 光谱总数 | {n_spectra} |
| 文件夹(组)总数 | {n_folders} |
| 待测物质 | 福美双(Thiram)、孔雀石绿(MG)、MBA (平等对待) |
| 主数据来源 | pure 主 benchmark（不混入 soil 外部验证） |
| 浓度体系 | 4/5/6 对应 10^-4 / 10^-5 / 10^-6 M |
| CV折数 | {N_FOLDS} |
| 结果命名空间 | {result_scope} |

### 混合类型分布
| 混合类型 | 光谱数 | 文件夹数 |
|----------|--------|----------|
"""
    for fam in sorted(meta['family'].unique()):
        sub = meta[meta['family'] == fam]
        cn = family_cn.get(fam, fam)
        report += f"| {cn} | {len(sub)} | {sub['folder_name'].nunique()} |\n"

    report += f"""
## 2. 任务定义
| 编号 | 目标列 | 类别 | 任务说明 |
|------|--------|------|----------|
"""
    for t in ordered_tasks:
        cn = task_cn.get(t['name'], t['name'])
        report += f"| {t['id']} | {t['col']} | {t['classes']} | {cn} |\n"

    report += f"""
## 3. 结果汇总 (Macro F1 ± 标准差)
"""
    models = sorted(agg_df['Model'].unique())
    report += "| 任务 | " + " | ".join(models) + " | 最佳模型 |\n"
    report += "|------|" + "|".join(["------"] * len(models)) + "|------|\n"

    for _, brow in best_df.iterrows():
        task_id = brow['Task']
        task_name = task_cn.get(brow['TaskName'], brow['TaskName'])
        sub = agg_df[agg_df['Task'] == task_id].set_index('Model')
        cells = []
        for m in models:
            if m in sub.index:
                cells.append(f"{sub.loc[m, 'F1_mean']:.3f}±{sub.loc[m, 'F1_std']:.3f}")
            else:
                cells.append("—")
        report += f"| {task_name} | " + " | ".join(cells) + f" | **{brow['Model']}** |\n"

    report += f"""
## 4. 关键发现

**发现1: 有无检测通常优于阳性样本摩尔分级。**
presence 任务通常比 positive-only molar grading 更稳定，这与预期一致：存在/缺失的判别边界通常强于相邻摩尔浓度等级之间的细粒度差异。

**发现2: 福美双仍可能是最容易的主组分。**
若福美双 presence 与 positive-only grading 同时占优，这通常意味着其在当前 AgNPs 条件下具有更稳定的可分辨光谱特征。

**发现3: 孔雀石绿摩尔分级通常仍是难点。**
如果 MG 的 positive-only grading 继续偏弱，优先从相邻浓度等级混淆、谱峰重叠和归一化后峰形差异不足这三方面解释。

**发现4: MBA 在部分组合中可能仍存在遮蔽效应。**
若 MBA 的 positive-only grading 仍弱于 presence，可从共存组分干扰和弱特征峰被覆盖的角度讨论。

**发现5: 假阳性与相邻等级错分都值得单独看混淆矩阵。**
presence 任务更关注假阳性/假阴性，positive-only grading 更关注相邻等级混淆，这两类错误不能混为一谈。

**发现6: 混合复杂度属于补充任务，不替代主结果。**
mixture complexity 适合作为结构性补充观察，但不应覆盖 presence + positive-only molar grading 这套主任务定义。

**发现7: 无单一模型在所有任务上占主导地位。**
不同模型家族很可能在不同 analyte / task 上各有优势，因此文章更应该突出系统 benchmark，而不是押注单模型通吃。

**发现8(初步): 深度学习模型(1D-CNN, 1D-ResNet)未优于传统ML基线。**
在当前样本规模下，传统 ML 仍可能与 DL/KAN 形成接近甚至更强的基线，这本身就是 benchmark 结果的一部分。

## 5. 图表说明
- `figures/{DATA_VERSION}/eda/`: EDA可视化 (均值光谱、PCA、划分分布等)
- `figures/{DATA_VERSION}/preprocessing/`: 预处理方案对比
- `figures/{DATA_VERSION}/models/`: 混淆矩阵、模型对比柱状图
"""

    report_path = REPORTS_DIR / f'evaluation_report_{DATA_VERSION}{report_scope_suffix}.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"  Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='SERS Multi-Component Analysis Pipeline')
    parser.add_argument('--step', choices=['data', 'eda', 'train', 'report', 'all'],
                        default='all', help='Which pipeline step to run')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild all caches')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model names to evaluate (default: all)')
    parser.add_argument('--preprocess', choices=list(PREPROCESS_TAGS),
                        default='p1', help='Preprocessing variant to use')
    parser.add_argument('--task-group', choices=['main', 'supplementary', 'all'],
                        default='main', help='Which task group to run')
    parser.add_argument('--split-file', default=ACTIVE_SPLIT_FILE,
                        help='Split CSV filename under the active versioned split directory (default from src.config.ACTIVE_SPLIT_FILE)')
    parser.add_argument('--result-scope', default='quick_check',
                        help='Result namespace under models; default quick_check saves to models/quick_check. Use benchmark scripts for formal grouped comparisons.')
    args = parser.parse_args()

    print(f"SERS Multi-Component Analysis Pipeline")
    print(f"Strategy 3: MBA as regular component")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")
    print(f"Data version: {DATA_VERSION}")
    print(f"Task profile: {TASK_PROFILE}")
    print(f"Split file: {args.split_file}")
    print(f"Result scope: {args.result_scope}")

    if args.step in ('data', 'all'):
        meta, wn, X_dict = step_data(args)
    else:
        # Load cached data
        import numpy as np
        import pandas as pd
        meta = pd.read_csv(SPLITS_DIR / args.split_file)
        # Fix boolean columns that may be read as strings from CSV
        for col in ['has_thiram', 'has_mg', 'has_mba']:
            if col in meta.columns:
                meta[col] = meta[col].astype(str).str.strip().str.lower().map(
                    {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
                ).fillna(0).astype(int)
        wn, X_dict = load_preprocessed_variants()

    if args.step in ('eda', 'all'):
        step_eda(meta, wn, X_dict)

    if args.step in ('train', 'all'):
        res_df, agg_df = step_train(meta, X_dict, args)
        predictions = res_df.attrs.get('predictions', {})

        if args.step in ('report', 'all'):
            step_report(meta, agg_df, predictions, result_scope=args.result_scope)
    elif args.step == 'report':
        # Load results from saved CSV files
        import pandas as pd
        tag = args.preprocess
        result_dir = resolve_result_dir(args.result_scope)
        summary_path = result_dir / f'cv_results_summary_{tag}.csv'
        if not summary_path.exists():
            print(f"ERROR: {summary_path} not found. Run --step train first.")
            return
        agg_df = pd.read_csv(summary_path)
        predictions = load_predictions(tag=tag, output_dir=result_dir)
        step_report(meta, agg_df, predictions, result_scope=args.result_scope)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
