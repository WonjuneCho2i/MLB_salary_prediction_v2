"""EDA 3-3: relate batting workload and performance to next-season salary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = [
    ("PA", "Plate appearances", 1.0),
    ("AVG", "Batting average", 1.0),
    ("home_run_rate", "Home-run rate (%)", 100.0),
    ("value_batting_per_600_pa", "Batting runs per 600 PA", 1.0),
    ("value_war", "WAR", 1.0),
    ("value_war_per_600_pa", "WAR per 600 PA", 1.0),
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "02_data_preprocessing" / "model_ready" / "training_model_ready.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def load_main_sample(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    sample = data.loc[data["train_eligible_main_pa50"].eq(1)].copy()
    required = ["target_salary_total", "log1p_target_salary", *[column for column, _, _ in METRICS]]
    sample[required] = sample[required].apply(pd.to_numeric, errors="coerce")
    if sample.empty or sample["target_salary_total"].lt(0).any():
        raise ValueError("The PA>=50 training sample is empty or contains a negative salary.")
    return sample


def salary_tick(log_value: float, _position: float) -> str:
    salary_millions = np.expm1(log_value) / 1_000_000
    if salary_millions < 0:
        return ""
    return f"${salary_millions:g}M"


def save_relationship_figure(sample: pd.DataFrame, path: Path) -> pd.DataFrame:
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), sharey=True, constrained_layout=True)
    rows: list[dict[str, object]] = []

    positive = sample.loc[sample["target_salary_total"].gt(0)].copy()
    salary_ticks_millions = np.array([0.5, 1, 2, 5, 10, 20, 40, 60])
    salary_ticks_log = np.log1p(salary_ticks_millions * 1_000_000)

    for axis, (column, label, multiplier) in zip(axes.flat, METRICS):
        panel = positive[[column, "log1p_target_salary"]].dropna().copy()
        panel["plot_x"] = panel[column] * multiplier
        pearson = panel["plot_x"].corr(panel["log1p_target_salary"])
        spearman = panel["plot_x"].rank().corr(panel["log1p_target_salary"].rank())

        sns.regplot(
            data=panel,
            x="plot_x",
            y="log1p_target_salary",
            ci=None,
            scatter_kws={"s": 13, "alpha": 0.20, "edgecolors": "none"},
            line_kws={"linewidth": 2.0, "color": "#b14b4b"},
            color="#4c78a8",
            ax=axis,
        )
        axis.set_title(f"{label}\nSpearman r={spearman:.2f}")
        axis.set_xlabel(label)
        axis.set_ylabel("")
        axis.set_yticks(salary_ticks_log)
        axis.yaxis.set_major_formatter(FuncFormatter(salary_tick))

        rows.append(
            {
                "feature": column,
                "display_label": label,
                "n": len(panel),
                "pearson_vs_log_salary": pearson,
                "spearman_vs_log_salary": spearman,
            }
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Next-season salary ($ millions; log scale)")
    figure.suptitle("Batting workload and performance vs. next-season salary (PA >= 50)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    sample = load_main_sample(args.input.resolve())
    correlations = save_relationship_figure(sample, figure_dir / "05_salary_vs_performance.png")
    correlations.to_csv(
        table_dir / "performance_salary_correlations.csv", index=False, encoding="utf-8-sig"
    )

    print("Performance relationship EDA complete")
    print(f"main_sample_rows: {len(sample)}")
    for row in correlations.itertuples(index=False):
        print(f"  {row.feature}: Spearman r={row.spearman_vs_log_salary:.4f}, n={row.n}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
