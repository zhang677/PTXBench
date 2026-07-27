#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


DEFAULT_AUDIT_CSV = Path(
    "/home/ubuntu/AccRL-exps/eval_runs/"
    "loop-sft-reconstruct-2026-0613-2014-20260615-004822/success_counts.csv"
)
DEFAULT_QWEN_EVAL = Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0610-2332-eval")
DEFAULT_START_EVAL = Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0611-1826-eval")
DEFAULT_OUT_CSV = Path("analysis/reconstruction-cases-vs-runs.csv")
DEFAULT_OUT_PNG = Path("analysis/reconstruction-cases-vs-runs.png")
DEFAULT_MHA_EVAL_DIRS = [
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0610-2332-prompt-v1-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0611-1826-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0613-2014-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0614-2257-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0615-0823-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0615-1652-mha"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0616-0048-mha"),
]
DEFAULT_MHA_OUT_CSV = Path("analysis/mha-correctness-vs-model-version.csv")


def count_success_kernels(eval_dir: Path) -> int:
    success_dir = eval_dir / "success"
    if not success_dir.is_dir():
        raise FileNotFoundError(f"missing success directory: {success_dir}")
    return sum(1 for _ in success_dir.glob("exp_*/kernel_v*.cu"))


def read_audit_rows(audit_csv: Path) -> list[dict[str, str]]:
    with audit_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def build_rows(audit_csv: Path, qwen_eval: Path, start_eval: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    rows.append(
        {
            "x": 0,
            "run_label": "qwen-27b eval\n2026-0610-2332",
            "run_date": "2026-0610-2332",
            "kind": "comparison_qwen_27b",
            "round": "",
            "eval_dir": str(qwen_eval),
            "success_count": count_success_kernels(qwen_eval),
            "source": "counted success/exp_*/kernel_v*.cu",
        }
    )
    rows.append(
        {
            "x": 1,
            "run_label": "start eval\n2026-0611-1826",
            "run_date": "2026-0611-1826",
            "kind": "starting_point",
            "round": "",
            "eval_dir": str(start_eval),
            "success_count": count_success_kernels(start_eval),
            "source": "counted success/exp_*/kernel_v*.cu",
        }
    )

    for offset, audit_row in enumerate(read_audit_rows(audit_csv), start=2):
        eval_dir = Path(audit_row["eval_dir"])
        counted = count_success_kernels(eval_dir)
        csv_count = int(audit_row["success_count"])
        if counted != csv_count:
            raise ValueError(
                f"{eval_dir}: audit CSV says {csv_count}, counted {counted}"
            )
        round_id = int(audit_row["round"])
        run_date = audit_row["run_date"]
        rows.append(
            {
                "x": offset,
                "run_label": f"reconstruct r{round_id}\n{run_date}",
                "run_date": run_date,
                "kind": "loop_sft_reconstruct",
                "round": round_id,
                "eval_dir": str(eval_dir),
                "success_count": counted,
                "source": str(audit_csv),
            }
        )

    return rows


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "x",
        "run_label",
        "run_date",
        "kind",
        "round",
        "success_count",
        "mha_success_count",
        "mha_eval_dir",
        "eval_dir",
        "source",
        "mha_source",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, object]], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    xs = [int(row["x"]) for row in rows]
    ys = [int(row["success_count"]) for row in rows]
    mha_ys = [int(row["mha_success_count"]) for row in rows]
    labels = [str(row["run_label"]) for row in rows]
    colors = [
        "#7a8a99" if row["kind"] == "comparison_qwen_27b" else
        "#4c78a8" if row["kind"] == "starting_point" else
        "#2f9e44"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    recon_line = ax.plot(
        xs,
        ys,
        color="#20252b",
        linewidth=1.8,
        alpha=0.72,
        zorder=1,
        label="Reconstruction success",
    )
    ax.scatter(xs, ys, s=115, c=colors, edgecolors="#20252b", linewidths=1.0, zorder=2)

    for x, y in zip(xs, ys):
        ax.annotate(
            str(y),
            xy=(x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2 = ax.twinx()
    mha_line = ax2.plot(
        xs,
        mha_ys,
        color="#8f6a2a",
        marker="s",
        markersize=7,
        linewidth=1.8,
        linestyle="--",
        label="MHA success",
        zorder=3,
    )
    for x, y in zip(xs, mha_ys):
        ax2.annotate(
            str(y),
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#6b4f1f",
        )

    ax.set_title("Reconstruction Success Cases vs Runs", fontsize=15, pad=14)
    ax.set_ylabel("Cases with success/exp_*/kernel_v*.cu")
    ax2.set_ylabel("MHA cases with success/exp_*/kernel_v*.cu", color="#6b4f1f")
    ax2.tick_params(axis="y", colors="#6b4f1f")
    ax.set_xlabel("Run sequence")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, max(ys) + 12)
    ax2.set_ylim(0, max(max(mha_ys, default=0) + 1, 1))
    ax.grid(axis="y", color="#d8dde3", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend(recon_line + mha_line, [line.get_label() for line in recon_line + mha_line], loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def add_mha_success_counts(rows: list[dict[str, object]], mha_eval_dirs: list[Path]) -> None:
    if len(rows) != len(mha_eval_dirs):
        raise ValueError(
            f"expected {len(rows)} MHA eval dirs to align with reconstruction rows, "
            f"got {len(mha_eval_dirs)}"
        )

    for row, mha_eval_dir in zip(rows, mha_eval_dirs):
        row["mha_eval_dir"] = str(mha_eval_dir)
        row["mha_success_count"] = count_success_kernels(mha_eval_dir)
        row["mha_source"] = "counted success/exp_*/kernel_v*.cu"


def write_mha_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["x", "run_label", "run_date", "mha_eval_dir", "mha_success_count", "mha_source"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reconstruction success cases across eval runs."
    )
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--qwen-eval", type=Path, default=DEFAULT_QWEN_EVAL)
    parser.add_argument("--start-eval", type=Path, default=DEFAULT_START_EVAL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument(
        "--mha-eval-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "MHA eval directory to include. May be passed multiple times. "
            "Defaults to known qwen36-27b MHA eval versions."
        ),
    )
    parser.add_argument("--mha-out-csv", type=Path, default=DEFAULT_MHA_OUT_CSV)
    args = parser.parse_args()

    rows = build_rows(args.audit_csv, args.qwen_eval, args.start_eval)
    add_mha_success_counts(rows, args.mha_eval_dir or DEFAULT_MHA_EVAL_DIRS)
    write_csv(rows, args.out_csv)
    write_plot(rows, args.out_png)
    write_mha_csv(rows, args.mha_out_csv)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_png}")
    print(f"wrote {args.mha_out_csv}")


if __name__ == "__main__":
    main()
