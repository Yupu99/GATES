#!/usr/bin/env python3
"""
Plot OpenAI-ES vs Guided-ES (surr1/5/10) across S/M/L with confidence bands.
Produces thesis-ready vector PDFs (tight bounding boxes, readable fonts).
Includes convergence, training/test total cost, and VM vs SLA trade-off plots.
"""

import argparse
import glob
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Thesis-friendly typography + tight exports
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 16,
        # Ensure saved figures are tightly cropped by default
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

BEST_LINE_RE = re.compile(
    r"^episode:\s*(\d+),\s*\[current_policy_population:\],\s*"
    r"best reward so far:\s*([-\d\.]+),\s*"
    r"best reward of the current generation:\s*([-\d\.]+)"
)
TRAIN_LINE_RE = re.compile(
    r"^episode:\s*(\d+),\s*\[the_basic_policy:\],\s*"
    r"current training reward:\s*([-\d\.]+),\s*"
    r"current training VM_cost:\s*([-\d\.]+),\s*"
    r"current training SLA_penalty:\s*([-\d\.]+)"
)
TEST_LINE_RE = re.compile(
    r"^episode:\s*(\d+),\s*\[<<<<----testing---->>>>\],\s*"
    r"current testing reward:\s*([-\d\.]+),\s*"
    r"current testing VM_cost:\s*([-\d\.]+),\s*"
    r"current testing SLA_penalty:\s*([-\d\.]+)"
)
META_RE = re.compile(
    r"\[SLURM\]\s+QUICK_COMPARE\s+variant=(\S+)\s+size=([SML])\s+seed=(\d+)"
)


def parse_log(path: str) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    variant = None
    size = None
    seed = None
    for line in lines:
        m = META_RE.search(line)
        if m:
            variant = m.group(1)
            size = m.group(2)
            seed = int(m.group(3))
            break
    if variant is None:
        return None, None

    rows: Dict[int, Dict[str, float]] = {}
    max_ep: Optional[int] = None

    for line in lines:
        line = line.strip()
        m = BEST_LINE_RE.match(line)
        if m:
            ep = int(m.group(1))
            rows.setdefault(ep, {})
            rows[ep]["best_reward_so_far"] = float(m.group(2))
            max_ep = ep if max_ep is None else max(max_ep, ep)
            continue

        m = TRAIN_LINE_RE.match(line)
        if m:
            ep = int(m.group(1))
            rows.setdefault(ep, {})
            rows[ep]["train_vm_cost"] = float(m.group(3))
            rows[ep]["train_sla_penalty"] = float(m.group(4))
            max_ep = ep if max_ep is None else max(max_ep, ep)
            continue

        m = TEST_LINE_RE.match(line)
        if m:
            ep = int(m.group(1))
            rows.setdefault(ep, {})
            rows[ep]["test_vm_cost"] = float(m.group(3))
            rows[ep]["test_sla_penalty"] = float(m.group(4))
            max_ep = ep if max_ep is None else max(max_ep, ep)
            continue

    data = []
    for ep in sorted(rows.keys()):
        row = rows[ep]
        train_total = None
        test_total = None
        train_reward = None
        test_reward = None

        if "train_vm_cost" in row and "train_sla_penalty" in row:
            train_total = row["train_vm_cost"] + row["train_sla_penalty"]
            train_reward = -train_total

        if "test_vm_cost" in row and "test_sla_penalty" in row:
            test_total = row["test_vm_cost"] + row["test_sla_penalty"]
            test_reward = -test_total

        data.append(
            {
                "variant": variant,
                "size": size,
                "seed": seed,
                "episode": ep,
                "best_reward_so_far": row.get("best_reward_so_far"),
                "train_vm_cost": row.get("train_vm_cost"),
                "train_sla_penalty": row.get("train_sla_penalty"),
                "test_vm_cost": row.get("test_vm_cost"),
                "test_sla_penalty": row.get("test_sla_penalty"),
                "train_total_cost": train_total,
                "test_total_cost": test_total,
                "train_reward": train_reward,
                "test_reward": test_reward,
            }
        )

    if not data:
        return None, max_ep
    return pd.DataFrame(data), max_ep


def load_logs(glob_patterns: list[str], min_ep: int) -> pd.DataFrame:
    files: list[str] = []
    for pattern in glob_patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_patterns}")

    dfs: list[pd.DataFrame] = []
    for p in files:
        df, max_ep = parse_log(p)
        if df is None or max_ep is None:
            continue
        if max_ep < min_ep:
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No logs reached episode {min_ep}")
    return pd.concat(dfs, ignore_index=True)


def _variant_pretty(v: str) -> str:
    # Nicer legend labels. Keeps mapping obvious.
    mapping = {
        "openai": "OpenAI-ES",
        "guided_surr1": "Guided-ES (S=1)",
        "guided_surr5": "Guided-ES (S=5)",
        "guided_surr10": "Guided-ES (S=10)",
    }
    return mapping.get(v, v)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    outpath: str,
    max_ep: int,
    tag: str = "",
) -> None:
    variants = ["openai", "guided_surr1", "guided_surr5", "guided_surr10"]
    sizes = ["S", "M", "L"]
    color_map = {
        "openai": "#d55e00",       # orange
        "guided_surr1": "#0072b2",  # blue
        "guided_surr5": "#009e73",  # green
        "guided_surr10": "#cc79a7", # purple
    }

    # Slightly taller to improve label readability in print
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)

    for ax, size in zip(axes, sizes):
        sub = df[(df["size"] == size) & df[metric].notna()]
        sub = sub[sub["episode"] <= max_ep]

        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue

            agg = vsub.groupby("episode")[metric].agg(["mean", "std"]).reset_index()
            x = agg["episode"].to_numpy()
            y = agg["mean"].to_numpy()
            s = agg["std"].fillna(0.0).to_numpy()

            ax.plot(
                x,
                y,
                label=_variant_pretty(v),
                color=color_map.get(v),
                linewidth=2.5,  # thicker for print
            )
            ax.fill_between(x, y - s, y + s, alpha=0.18, color=color_map.get(v))

        ax.set_title(f"WF size {size}")
        ax.set_xlabel("Generation")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    axes[0].set_ylabel(ylabel)

    # One shared legend; reserve space for it explicitly (prevents awkward whitespace)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    # Reserve bottom space for legend; keep title inside figure bounds
    fig.subplots_adjust(bottom=0.22)
    full_title = f"{title}{' [' + tag + ']' if tag else ''}"
    fig.suptitle(full_title, y=0.98)

    # Save as vector PDF with tight bounding box
    fig.savefig(outpath)
    plt.close(fig)


def plot_vm_sla_tradeoff(
    df: pd.DataFrame,
    outpath: str,
    max_ep: int,
    tag: str = "",
    debug: bool = False,
) -> None:
    variants = ["openai", "guided_surr1", "guided_surr5", "guided_surr10"]
    sizes = ["S", "M", "L"]
    color_map = {
        "openai": "#d55e00",
        "guided_surr1": "#0072b2",
        "guided_surr5": "#009e73",
        "guided_surr10": "#cc79a7",
    }

    stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for size in sizes:
        sub = df[
            (df["size"] == size)
            & df["test_vm_cost"].notna()
            & df["test_sla_penalty"].notna()
        ].copy()
        if sub.empty:
            if debug:
                print(f"[vm_sla] size {size}: no test VM/SLA rows (before ep filter)")
            continue

        sub["episode"] = pd.to_numeric(sub["episode"], errors="coerce")
        sub = sub.dropna(subset=["episode"])
        sub = sub[sub["episode"] <= max_ep]
        if sub.empty:
            if debug:
                print(f"[vm_sla] size {size}: no test VM/SLA rows with episode <= {max_ep}")
            continue

        for v in variants:
            vsub_all = sub[sub["variant"] == v]
            if vsub_all.empty:
                if debug:
                    print(f"[vm_sla] size {size} variant {v}: no rows")
                continue

            v_ep = int(vsub_all["episode"].max())
            vsub = vsub_all[vsub_all["episode"] == v_ep]
            if vsub.empty:
                if debug:
                    print(f"[vm_sla] size {size} variant {v}: no rows at ep {v_ep}")
                continue

            vm = vsub["test_vm_cost"].to_numpy()
            sla = vsub["test_sla_penalty"].to_numpy()

            vm_mean, sla_mean = float(np.mean(vm)), float(np.mean(sla))
            vm_std = float(np.std(vm, ddof=1)) if len(vm) > 1 else 0.0
            sla_std = float(np.std(sla, ddof=1)) if len(sla) > 1 else 0.0

            stats[(size, v)] = {
                "vm_mean": vm_mean,
                "sla_mean": sla_mean,
                "vm_std": vm_std,
                "sla_std": sla_std,
                "n": float(len(vm)),
                "ep": float(v_ep),
            }
            if debug:
                print(
                    f"[vm_sla] size {size} variant {v} ep {v_ep}: "
                    f"n={len(vm)} vm_mean={vm_mean:.4f} sla_mean={sla_mean:.4f}"
                )

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.8))

    x = np.arange(len(sizes))
    group_width = 0.9
    variant_slot = group_width / max(len(variants), 1)
    bar_width = variant_slot * 0.42
    vm_offset = -bar_width * 0.55
    sla_offset = bar_width * 0.55

    variant_handles = []
    variant_labels = []

    for i, v in enumerate(variants):
        for si, size in enumerate(sizes):
            info = stats.get((size, v))
            if info is None:
                continue

            center = x[si] - group_width / 2 + (i + 0.5) * variant_slot
            vm_x = center + vm_offset
            sla_x = center + sla_offset

            vm_mean = info["vm_mean"]
            sla_mean = info["sla_mean"]
            vm_std = info["vm_std"]
            sla_std = info["sla_std"]
            color = color_map.get(v)

            container = ax.bar(
                vm_x,
                vm_mean,
                width=bar_width,
                color=color,
                alpha=0.9,
                label=_variant_pretty(v) if si == 0 else None,
            )
            ax.bar(
                sla_x,
                sla_mean,
                width=bar_width,
                color="white",
                edgecolor=color,
                hatch="///",
                linewidth=1.0,
            )

            if si == 0:
                variant_handles.append(container.patches[0])
                variant_labels.append(_variant_pretty(v))

            if vm_std > 0:
                ax.errorbar(
                    vm_x,
                    vm_mean,
                    yerr=vm_std,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2,
                    zorder=5,
                )
            if sla_std > 0:
                ax.errorbar(
                    sla_x,
                    sla_mean,
                    yerr=sla_std,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2,
                    zorder=5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"WF size {s}" for s in sizes])
    ax.set_ylabel("Test cost (log scale)")
    ax.set_yscale("log")

    from matplotlib.ticker import FuncFormatter, LogLocator

    min_pos = None
    max_pos = None
    for info in stats.values():
        candidates = [
            info["vm_mean"],
            info["sla_mean"],
            info["vm_mean"] - info["vm_std"],
            info["sla_mean"] - info["sla_std"],
            info["vm_mean"] + info["vm_std"],
            info["sla_mean"] + info["sla_std"],
        ]
        for val in candidates:
            if val is None or val <= 0:
                continue
            min_pos = val if min_pos is None else min(min_pos, val)
            max_pos = val if max_pos is None else max(max_pos, val)

    if min_pos is None or max_pos is None:
        min_pos, max_pos = 1.0, 10.0

    y_min = min_pos * 0.6
    y_max = max_pos * 1.4
    if y_min <= 0:
        y_min = min_pos * 0.5

    ax.set_ylim(bottom=y_min, top=y_max)

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{int(y)}" if y >= 1 else f"{y:g}")
    )

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    from matplotlib.patches import Patch

    component_handles = [
        Patch(facecolor="#6e6e6e", edgecolor="#6e6e6e", label="VM cost"),
        Patch(
            facecolor="white",
            edgecolor="#6e6e6e",
            hatch="///",
            linewidth=1.0,
            label="SLA penalty",
        ),
    ]

    component_legend = ax.legend(
        component_handles,
        [h.get_label() for h in component_handles],
        loc="upper left",
        frameon=False,
    )
    ax.add_artist(component_legend)
    ax.legend(
        variant_handles,
        variant_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=4,
    )

    full_title = ( 
        ""
        + (f" [{tag}]" if tag else "")
    )
    fig.suptitle(full_title, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compare results across S/M/L.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern for logs; can be passed multiple times",
    )
    parser.add_argument("--min_ep", type=int, default=0)
    parser.add_argument("--max_ep", type=int, default=500)
    parser.add_argument("--outdir", default="results_plots/compare")
    parser.add_argument("--tag", default="", help="Optional label suffix for output filenames (e.g., best_hyper)")
    
    args = parser.parse_args()

    glob_patterns = args.glob if args.glob else ["slurm_compare_*.out"]
    df = load_logs(glob_patterns, args.min_ep)
    os.makedirs(args.outdir, exist_ok=True)

    tag = args.tag.strip()
    suffix = f"_{tag}" if tag else ""

    plot_metric(
        df,
        metric="best_reward_so_far",
        ylabel="Best reward so far",
        title="Training convergence",
        outpath=os.path.join(args.outdir, f"convergence_best_reward{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
    )
    plot_metric(
        df,
        metric="train_total_cost",
        ylabel="Train total cost (VM + SLA)",
        title="Training total cost",
        outpath=os.path.join(args.outdir, f"training_total_cost{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
    )
    plot_metric(
        df,
        metric="test_total_cost",
        ylabel="Test total cost (VM + SLA)",
        title="Testing total cost",
        outpath=os.path.join(args.outdir, f"testing_total_cost{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
    )
    plot_metric(
        df,
        metric="train_reward",
        ylabel="Train reward (-(VM + SLA))",
        title="Training reward",
        outpath=os.path.join(args.outdir, f"training_reward{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
    )
    plot_metric(
        df,
        metric="test_reward",
        ylabel="Test reward (-(VM + SLA))",
        title="Testing reward",
        outpath=os.path.join(args.outdir, f"testing_reward{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
    )
    plot_vm_sla_tradeoff(
        df,
        outpath=os.path.join(args.outdir, f"vm_sla_tradeoff{suffix}.pdf"),
        max_ep=args.max_ep,
        tag=tag,
        debug=args.vm_sla_debug,
    )

    print(f"Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
