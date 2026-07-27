#!/usr/bin/env python3
"""Plot trajectory correctness rate against model release date.

For each experiment listed in experiments.csv, this uses either the first N
sorted trajectory files, where N is the minimum trajectory count across all
listed experiments, or all trajectories per experiment when requested. For each
turn limit T, a trajectory is counted correct if any of its first T evaluation
turns is classified as Correct.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from export_turn_correctness_arch import as_number, extract_turn_sequence, load_experiment_rows


MODEL_RELEASE_DATES = Path(__file__).resolve().parent / "model_release_dates.csv"
PTX_RELEASE_DATES = Path(__file__).resolve().parent / "ptx_release_dates.csv"
FA_RELEASE_DATES = Path(__file__).resolve().parent / "fa_release_dates.csv"
WORKLOADS_DIR = Path("/home/ubuntu/accrl-training/workloads")
DEFINITIONS_DIR = Path("/home/ubuntu/accrl-training/definitions")
DEFAULT_MIGRATION_PATH = Path(__file__).resolve().parent / "migration.csv"
DEFAULT_TURN_LIMITS = [1, 4, 8]
DEFAULT_XLIM = (datetime(2022, 7, 1), datetime(2026, 7, 1))
TURN_MARKERS = {
    1: "o",
    4: "s",
    8: "^",
}
ARCH_TAG_BY_ARCH = {
    "hopper": "H",
    "blackwell": "B",
}
MODEL_COLOR_PALETTE = [
    "#376B9D",
    "#B45F06",
    "#2F6B5F",
    "#7A5195",
    "#D1495B",
    "#4D908E",
    "#8F6A00",
    "#5A5A5A",
    "#C75146",
    "#2D5D7B",
]


def sorted_model_names(models) -> list[str]:
    return sorted(set(models), key=lambda value: value.casefold())


def build_model_color_map(models) -> dict[str, str]:
    return {
        model: MODEL_COLOR_PALETTE[idx % len(MODEL_COLOR_PALETTE)]
        for idx, model in enumerate(sorted_model_names(models))
    }


def parse_date(value: str, allow_month_day_overflow: bool = False) -> datetime:
    value = value.strip()
    for fmt in ("%Y-%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    if allow_month_day_overflow and len(value) == 9 and value[4] == "-":
        year = int(value[:4])
        month = int(value[5:7])
        day = int(value[7:9])
        if 1 <= month <= 12 and day >= 1:
            return datetime(year, month, 1) + timedelta(days=day - 1)
    raise ValueError(f"Unsupported date format: {value!r}")


def load_release_dates(
    path: Path,
    key_column: str,
    date_columns: tuple[str, ...] = ("date",),
    allow_month_day_overflow: bool = False,
) -> dict[str, datetime]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        date_column = next((column for column in date_columns if column in fieldnames), None)
        if key_column not in fieldnames or date_column is None:
            raise ValueError(
                f"{path} must contain {key_column!r} and one of {date_columns!r} columns"
            )
        return {
            row[key_column]: parse_date(row[date_column], allow_month_day_overflow)
            for row in reader
        }


def load_fa_release_dates(path: Path) -> dict[str, tuple[str, datetime]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        date_column = next((column for column in ("date", "dates") if column in fieldnames), None)
        required = {"version", "arch"}
        missing = required.difference(fieldnames)
        if missing or date_column is None:
            raise ValueError(
                f"{path} must contain version, arch, and one of ('date', 'dates') columns"
            )

        release_dates = {}
        for row in reader:
            release_dates[row["arch"].strip().lower()] = (
                row["version"].strip(),
                parse_date(row[date_column], allow_month_day_overflow=True),
            )
        return release_dates


def release_date_for_model(model_dates: dict[str, datetime], model: str) -> datetime:
    try:
        return model_dates[model]
    except KeyError as exc:
        raise KeyError(f"Missing release date for model {model!r}") from exc


def axis_value_label(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in text)


def build_workload_name(
    definition_name: str,
    axes: dict,
    var_axis_names: list[str] | tuple[str, ...],
) -> str:
    axis_parts = [
        f"{axis_name}{axis_value_label(axis_value)}"
        for axis_name in var_axis_names
        if axis_name in axes
        for axis_value in [axes[axis_name]]
    ]
    if not axis_parts:
        return definition_name
    return f"{definition_name}_{'_'.join(axis_parts)}"


def load_definition_var_axes(definitions_dir: Path) -> dict[str, list[str]]:
    definition_var_axes: dict[str, list[str]] = {}
    for definition_path in sorted(definitions_dir.glob("*/*.json")):
        try:
            definition = json.loads(definition_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {definition_path}") from exc

        definition_name = definition.get("name") or definition_path.stem
        axes = definition.get("axes", {})
        if not isinstance(axes, dict):
            continue
        definition_var_axes[definition_name] = [
            axis_name
            for axis_name, axis_info in axes.items()
            if isinstance(axis_info, dict) and axis_info.get("type") == "var"
        ]
    return definition_var_axes


def load_workload_index(
    workloads_dir: Path,
    definition_var_axes: dict[str, list[str]],
) -> dict[str, list[dict[str, str]]]:
    workload_index: dict[str, list[dict[str, str]]] = {}
    for jsonl_path in sorted(workloads_dir.glob("*/*.jsonl")):
        with jsonl_path.open() as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {jsonl_path}:{line_number}") from exc

                workload = record.get("workload", {})
                workload_id = workload.get("uuid")
                definition_name = record.get("definition", "")
                if not workload_id or not definition_name:
                    continue

                axes = workload.get("axes", {})
                workload_index.setdefault(workload_id, []).append(
                    {
                        "definition": definition_name,
                        "workload_name": build_workload_name(
                            definition_name,
                            axes,
                            definition_var_axes.get(definition_name, []),
                        ),
                    }
                )
    return workload_index


def load_workload_migrations(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}

    rows_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"old_definition", "old_workload", "new_definition", "new_workload"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            old_definition = row["old_definition"].strip()
            old_workload = row["old_workload"].strip()
            new_definition = row["new_definition"].strip()
            new_workload = row["new_workload"].strip()
            if not (old_definition and old_workload and new_definition and new_workload):
                continue
            rows_by_key.setdefault((old_definition, old_workload), []).append(
                {
                    "new_definition": new_definition,
                    "new_workload": new_workload,
                    "seq": row.get("seq", "").strip(),
                }
            )

    return {key: rows[0] for key, rows in rows_by_key.items() if len(rows) == 1}


def select_workload_entry(
    row: dict[str, str],
    workload_index: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    workload_id = row.get("workload", "").strip() or row.get("workload_uuid", "").strip()
    definition = row.get("definition", "").strip()
    if not workload_id or not definition:
        return None

    entries = workload_index.get(workload_id, [])
    for entry in entries:
        if entry["definition"] == definition:
            return entry
    return None


def apply_workload_migration(
    row: dict[str, str],
    workload_migrations: dict[tuple[str, str], dict[str, str]],
) -> None:
    definition = row.get("definition", "").strip()
    workload = row.get("workload", "").strip() or row.get("workload_uuid", "").strip()
    migration = workload_migrations.get((definition, workload))
    if migration is None:
        return

    row["definition"] = migration["new_definition"]
    row["workload"] = migration["new_workload"]
    if "workload_uuid" in row:
        row["workload_uuid"] = migration["new_workload"]
    if migration.get("seq"):
        row["migration_seq"] = migration["seq"]
    row.pop("workload_name", None)


def enrich_experiment_rows(
    experiment_rows: list[dict[str, str]],
    workload_index: dict[str, list[dict[str, str]]],
    workload_migrations: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    enriched_rows = []
    for row in experiment_rows:
        enriched = dict(row)

        apply_workload_migration(enriched, workload_migrations)
        workload_entry = select_workload_entry(enriched, workload_index)
        workload_id = (
            enriched.get("workload", "").strip()
            or enriched.get("workload_uuid", "").strip()
        )

        if workload_entry is not None:
            enriched["workload_name"] = workload_entry["workload_name"]
        elif enriched.get("migration_seq"):
            enriched["workload_name"] = (
                f"{enriched.get('definition', '').strip()}_S{enriched['migration_seq']}"
            )
        elif not enriched.get("workload_name"):
            enriched["workload_name"] = (
                enriched.get("definition", "").strip()
                or workload_id
                or "unknown_workload"
            )

        enriched_rows.append(enriched)
    return enriched_rows


def trajectory_paths(run_dir: Path) -> list[Path]:
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing trajectories directory: {traj_dir}")
    paths = sorted(traj_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No trajectory JSON files found in {traj_dir}")
    return paths


def trajectory_sequence(path: Path) -> list[str]:
    traj = json.loads(path.read_text())
    return extract_turn_sequence(traj)


def select_trajectory_paths(
    experiment_rows: list[dict[str, str]],
    sample_mode: str,
) -> tuple[int | None, list[tuple[dict[str, str], list[Path]]]]:
    path_sets = [(row, trajectory_paths(Path(row["exp_dir"]))) for row in experiment_rows]
    min_trajectories = min(len(paths) for _row, paths in path_sets)
    selected_sets = []
    for row, paths in path_sets:
        if sample_mode == "all":
            selected = paths
        else:
            selected = paths[:min_trajectories]
        selected_sets.append((row, selected))
    return None if sample_mode == "all" else min_trajectories, selected_sets


def compute_rates(
    experiment_rows: list[dict[str, str]],
    model_dates: dict[str, datetime],
    turn_limits: list[int],
    sample_mode: str,
) -> tuple[int | None, list[dict]]:
    min_trajectories, selected_sets = select_trajectory_paths(experiment_rows, sample_mode)

    results = []
    for row, selected in selected_sets:
        model = row["model"]
        date = release_date_for_model(model_dates, model)

        sequences = [trajectory_sequence(path) for path in selected]
        total = len(selected)
        correct_by_t = {
            turn_limit: sum(1 for seq in sequences if "Correct" in seq[:turn_limit])
            for turn_limit in turn_limits
        }
        results.append(
            {
                "model": model,
                "arch": row["arch"],
                "workload": row.get("workload", ""),
                "workload_name": row.get("workload_name", ""),
                "definition": row.get("definition", ""),
                "exp_dir": row["exp_dir"],
                "date": date,
                "n_trajectories": total,
                "correct_by_t": correct_by_t,
                "correctness_rate_by_t": {
                    turn_limit: correct / total if total else 0.0
                    for turn_limit, correct in correct_by_t.items()
                },
            }
        )

    return min_trajectories, results


def load_turn_speedups(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    csv_path = run_dir / "figures" / "turn_correctness_arch.csv"
    if not csv_path.exists():
        return {}

    speedups: dict[str, list[tuple[int, float]]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turn = as_number(row.get("turn"))
            speedup = as_number(row.get("speedup"))
            if turn is None or speedup is None:
                continue
            speedups.setdefault(row["trajectory_id"], []).append((int(turn), speedup))
    return speedups


def best_speedup_from_turn_rows(
    turn_rows: list[tuple[int, float]],
    turn_limit: int,
) -> float:
    return max(
        (speedup for turn, speedup in turn_rows if turn < turn_limit),
        default=0.0,
    )


def compute_best_speedup_results(
    experiment_rows: list[dict[str, str]],
    model_dates: dict[str, datetime],
    turn_limits: list[int],
    sample_mode: str,
) -> tuple[int | None, list[dict]]:
    min_trajectories, selected_sets = select_trajectory_paths(experiment_rows, sample_mode)

    results = []
    for row, selected in selected_sets:
        model = row["model"]
        date = release_date_for_model(model_dates, model)

        run_dir = Path(row["exp_dir"])
        turn_speedups = load_turn_speedups(run_dir)
        best_speedup_by_t = {
            turn_limit: max(
                (
                    best_speedup_from_turn_rows(turn_speedups.get(path.stem, []), turn_limit)
                    for path in selected
                ),
                default=0.0,
            )
            for turn_limit in turn_limits
        }
        results.append(
            {
                "model": model,
                "arch": row["arch"],
                "workload": row.get("workload", ""),
                "workload_name": row.get("workload_name", ""),
                "definition": row.get("definition", ""),
                "exp_dir": row["exp_dir"],
                "date": date,
                "n_trajectories": len(selected),
                "best_speedup_by_t": best_speedup_by_t,
            }
        )

    return min_trajectories, results


def load_arch_tags(run_dir: Path) -> dict[str, str]:
    csv_path = run_dir / "figures" / "turn_correctness_arch.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing exported arch CSV: {csv_path}")

    tags = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory_id = row["trajectory_id"]
            arch_tag = row.get("arch_tag", "")
            if trajectory_id not in tags or arch_tag:
                tags[trajectory_id] = arch_tag
    return tags


def compute_arch_instruction_results(
    experiment_rows: list[dict[str, str]],
    model_dates: dict[str, datetime],
    turn_limits: list[int],
    sample_mode: str,
) -> tuple[int | None, list[dict]]:
    min_trajectories, selected_sets = select_trajectory_paths(experiment_rows, sample_mode)

    results = []
    for row, selected in selected_sets:
        model = row["model"]
        date = release_date_for_model(model_dates, model)

        arch = row["arch"].strip().lower()
        required_tag = ARCH_TAG_BY_ARCH.get(arch)
        if required_tag is None:
            raise ValueError(f"Unsupported architecture for arch-tag metric: {row['arch']!r}")

        run_dir = Path(row["exp_dir"])
        arch_tags = load_arch_tags(run_dir)
        samples = [
            (trajectory_sequence(path), arch_tags.get(path.stem, ""))
            for path in selected
        ]
        total = len(samples)
        correct_with_arch_by_t = {
            turn_limit: sum(
                1
                for seq, arch_tag in samples
                if "Correct" in seq[:turn_limit]
                and required_tag in [tag.strip() for tag in arch_tag.split(",")]
            )
            for turn_limit in turn_limits
        }
        results.append(
            {
                "model": model,
                "arch": row["arch"],
                "workload": row.get("workload", ""),
                "workload_name": row.get("workload_name", ""),
                "definition": row.get("definition", ""),
                "exp_dir": row["exp_dir"],
                "date": date,
                "n_trajectories": total,
                "required_arch_tag": required_tag,
                "correct_with_arch_by_t": correct_with_arch_by_t,
                "correct_with_arch_rate_by_t": {
                    turn_limit: count / total if total else 0.0
                    for turn_limit, count in correct_with_arch_by_t.items()
                },
            }
        )

    return min_trajectories, results


def write_summary_csv(results: list[dict], turn_limits: list[int], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "arch",
            "workload",
            "workload_name",
            "definition",
            "exp_dir",
            "date",
            "turn_limit",
            "n_trajectories",
            "correct",
            "correctness_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for turn_limit in turn_limits:
                writer.writerow(
                    {
                        "model": row["model"],
                        "arch": row["arch"],
                        "workload": row["workload"],
                        "workload_name": row["workload_name"],
                        "definition": row["definition"],
                        "exp_dir": row["exp_dir"],
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "turn_limit": turn_limit,
                        "n_trajectories": row["n_trajectories"],
                        "correct": row["correct_by_t"][turn_limit],
                        "correctness_rate": f"{row['correctness_rate_by_t'][turn_limit]:.6f}",
                    }
                )


def write_best_speedup_csv(results: list[dict], turn_limits: list[int], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "arch",
            "workload",
            "workload_name",
            "definition",
            "exp_dir",
            "date",
            "turn_limit",
            "n_trajectories",
            "best_speedup",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for turn_limit in turn_limits:
                writer.writerow(
                    {
                        "model": row["model"],
                        "arch": row["arch"],
                        "workload": row["workload"],
                        "workload_name": row["workload_name"],
                        "definition": row["definition"],
                        "exp_dir": row["exp_dir"],
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "turn_limit": turn_limit,
                        "n_trajectories": row["n_trajectories"],
                        "best_speedup": f"{row['best_speedup_by_t'][turn_limit]:.9g}",
                    }
                )


def write_arch_instruction_csv(results: list[dict], turn_limits: list[int], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "arch",
            "required_arch_tag",
            "workload",
            "workload_name",
            "definition",
            "exp_dir",
            "date",
            "turn_limit",
            "n_trajectories",
            "correct_with_arch",
            "correct_with_arch_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for turn_limit in turn_limits:
                writer.writerow(
                    {
                        "model": row["model"],
                        "arch": row["arch"],
                        "required_arch_tag": row["required_arch_tag"],
                        "workload": row["workload"],
                        "workload_name": row["workload_name"],
                        "definition": row["definition"],
                        "exp_dir": row["exp_dir"],
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "turn_limit": turn_limit,
                        "n_trajectories": row["n_trajectories"],
                        "correct_with_arch": row["correct_with_arch_by_t"][turn_limit],
                        "correct_with_arch_rate": f"{row['correct_with_arch_rate_by_t'][turn_limit]:.6f}",
                    }
                )


def sample_label(sample_mode: str, min_trajectories: int | None) -> str:
    if sample_mode == "all":
        return "all trajectories"
    return f"first {min_trajectories} trajectories"


def slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "unknown"


def grouped_by_workload_arch(results: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for row in results:
        key = (row.get("workload_name", ""), row["arch"].strip().lower())
        grouped.setdefault(key, []).append(row)
    return grouped


def workload_title(row: dict) -> str:
    return row.get("workload_name") or row.get("definition") or "unknown_workload"


def workload_arch_title(row: dict) -> str:
    arch = row.get("arch", "").strip().lower()
    if not arch:
        return workload_title(row)
    return f"{workload_title(row)} / {arch}"


def fa_reference_lines(
    definition: str,
    arch: str,
    fa_dates: dict[str, tuple[str, datetime]],
) -> list[tuple[datetime, str, str, str]]:
    if "mha" not in definition.lower():
        return []
    fa_release = fa_dates.get(arch.strip().lower())
    if fa_release is None:
        return []
    version, date = fa_release
    return [(date, f"{version} release ({date:%Y-%m-%d})", "#00876C", "--")]


def ptx_reference_line(
    arch: str,
    ptx_dates: dict[str, datetime],
) -> tuple[datetime, str, str, str]:
    arch = arch.strip().lower()
    if arch not in ptx_dates:
        raise KeyError(f"Missing PTX release date for arch {arch!r}")
    date = ptx_dates[arch]
    return (date, f"{arch} PTX release ({date:%Y-%m-%d})", "#8A3FFC", ":")


def plot_turn_limit_metric(
    results: list[dict],
    min_trajectories: int | None,
    sample_mode: str,
    turn_limits: list[int],
    out_path: Path,
    value_by_turn_key: str,
    title_metric: str,
    y_label: str,
    percent_y: bool,
    y_limit: tuple[float, float] | None = None,
    reference_lines: list[tuple[datetime, str, str, str]] | None = None,
    title_context: str = "",
    model_colors: dict[str, str] | None = None,
) -> None:
    results = sorted(results, key=lambda row: row["date"])
    reference_lines = reference_lines or []
    model_colors = model_colors or build_model_color_map(row["model"] for row in results)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for row in results:
        ax.plot(
            [row["date"]] * len(turn_limits),
            [row[value_by_turn_key][turn_limit] for turn_limit in turn_limits],
            color=model_colors[row["model"]],
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            zorder=2,
        )

    for turn_limit in turn_limits:
        dates = [row["date"] for row in results]
        rates = [row[value_by_turn_key][turn_limit] for row in results]
        colors = [model_colors[row["model"]] for row in results]
        ax.scatter(
            dates,
            rates,
            s=78,
            marker=TURN_MARKERS.get(turn_limit, "o"),
            color=colors,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    label_turn_limit = max(turn_limits)
    for row in results:
        ax.annotate(
            f"({row['n_trajectories']})",
            (row["date"], row[value_by_turn_key][label_turn_limit]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color=model_colors[row["model"]],
            ha="left",
            va="bottom",
            zorder=4,
        )

    for date, label, color, linestyle in reference_lines:
        ax.axvline(
            date,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    title = f"{title_metric} by Model Release Date"
    if title_context:
        title = f"{title}\n{title_context}"
    ax.set_title(f"{title} ({sample_label(sample_mode, min_trajectories)})")
    ax.set_xlabel("Release date")
    ax.set_ylabel(y_label)
    ax.set_xlim(DEFAULT_XLIM)
    values = [
        row[value_by_turn_key][turn_limit]
        for row in results
        for turn_limit in turn_limits
    ]
    if percent_y:
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{y:.0%}"))
    elif y_limit is not None:
        ax.set_ylim(y_limit)
    else:
        max_value = max(values, default=0.0)
        ax.set_ylim(0, max(1.0, max_value * 1.12))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)

    result_models = sorted_model_names(row["model"] for row in results)
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=model_colors[model],
            markeredgecolor="white",
            markersize=8,
            label=model,
        )
        for model in result_models
    ]
    turn_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=TURN_MARKERS.get(turn_limit, "o"),
            color="#555555",
            linestyle="none",
            markerfacecolor="#555555",
            markersize=8,
            label=f"T={turn_limit}",
        )
        for turn_limit in turn_limits
    ]
    reference_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        for _date, label, color, linestyle in reference_lines
    ]
    ax.legend(
        handles=model_handles + turn_handles + reference_handles,
        loc="best",
        fontsize=8,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_results(
    results: list[dict],
    min_trajectories: int | None,
    sample_mode: str,
    turn_limits: list[int],
    out_path: Path,
    reference_lines: list[tuple[datetime, str, str, str]] | None = None,
    title_context: str = "",
    model_colors: dict[str, str] | None = None,
) -> None:
    plot_turn_limit_metric(
        results=results,
        min_trajectories=min_trajectories,
        sample_mode=sample_mode,
        turn_limits=turn_limits,
        out_path=out_path,
        value_by_turn_key="correctness_rate_by_t",
        title_metric="Correctness Rate",
        y_label="Correctness rate",
        percent_y=True,
        reference_lines=reference_lines,
        title_context=title_context,
        model_colors=model_colors,
    )


def plot_best_speedup_by_release_date(
    results: list[dict],
    min_trajectories: int | None,
    sample_mode: str,
    turn_limits: list[int],
    out_path: Path,
    reference_lines: list[tuple[datetime, str, str, str]] | None = None,
    title_context: str = "",
    model_colors: dict[str, str] | None = None,
) -> None:
    plot_turn_limit_metric(
        results=results,
        min_trajectories=min_trajectories,
        sample_mode=sample_mode,
        turn_limits=turn_limits,
        out_path=out_path,
        value_by_turn_key="best_speedup_by_t",
        title_metric="Best Speedup",
        y_label="Best speedup",
        percent_y=False,
        y_limit=(0, 1.0),
        reference_lines=reference_lines,
        title_context=title_context,
        model_colors=model_colors,
    )


def plot_arch_instruction_correctness_by_release_date(
    results: list[dict],
    min_trajectories: int | None,
    sample_mode: str,
    turn_limits: list[int],
    out_path: Path,
    reference_lines: list[tuple[datetime, str, str, str]] | None = None,
    title_context: str = "",
    model_colors: dict[str, str] | None = None,
) -> None:
    plot_turn_limit_metric(
        results=results,
        min_trajectories=min_trajectories,
        sample_mode=sample_mode,
        turn_limits=turn_limits,
        out_path=out_path,
        value_by_turn_key="correct_with_arch_rate_by_t",
        title_metric="Correctness Rate with Architecture Instructions",
        y_label="Correct and architecture-tagged rate",
        percent_y=True,
        reference_lines=reference_lines,
        title_context=title_context,
        model_colors=model_colors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments.csv",
        help="CSV with model, arch, definition, workload, and exp_dir columns",
    )
    parser.add_argument(
        "--model-release-dates",
        type=Path,
        default=MODEL_RELEASE_DATES,
        help="CSV with model,date columns",
    )
    parser.add_argument(
        "--ptx-release-dates",
        type=Path,
        default=PTX_RELEASE_DATES,
        help="CSV with arch,date columns",
    )
    parser.add_argument(
        "--fa-release-dates",
        type=Path,
        default=FA_RELEASE_DATES,
        help="CSV with version,arch,date/dates columns for FlashAttention reference lines",
    )
    parser.add_argument(
        "--workloads-dir",
        type=Path,
        default=WORKLOADS_DIR,
        help="Directory containing workload JSONL files used to resolve workload UUIDs",
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=DEFINITIONS_DIR,
        help="Directory containing definition JSON files used to identify variable axes",
    )
    parser.add_argument(
        "--migration-csv",
        type=Path,
        default=DEFAULT_MIGRATION_PATH,
        help="CSV-style migration.csv mapping old definition/workload to new definition/workload",
    )
    parser.add_argument(
        "--turn-limits",
        type=int,
        nargs="+",
        default=DEFAULT_TURN_LIMITS,
        help="Turn limits used for correctness rates",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiments-csv parent>/figures)",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write summary CSVs and skip grouped PNG generation",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "all"],
        default="all",
        help="balanced uses the first minimum trajectory count across experiments; all uses every trajectory per experiment",
    )
    args = parser.parse_args()

    definition_var_axes = load_definition_var_axes(args.definitions_dir)
    workload_index = load_workload_index(args.workloads_dir, definition_var_axes)
    workload_migrations = load_workload_migrations(args.migration_csv)
    experiment_rows = enrich_experiment_rows(
        load_experiment_rows(args.experiments_csv),
        workload_index,
        workload_migrations,
    )
    required = {"model", "arch", "definition", "workload", "exp_dir"}
    missing = required.difference(experiment_rows[0].keys() if experiment_rows else [])
    if missing:
        raise ValueError(
            f"{args.experiments_csv} is missing required columns: {', '.join(sorted(missing))}"
        )
    model_colors = build_model_color_map(row["model"] for row in experiment_rows)

    model_dates = load_release_dates(args.model_release_dates, "model")

    turn_limits = sorted(args.turn_limits)
    min_trajectories, results = compute_rates(
        experiment_rows,
        model_dates,
        turn_limits,
        args.sample_mode,
    )
    speedup_min_trajectories, speedup_results = compute_best_speedup_results(
        experiment_rows,
        model_dates,
        turn_limits,
        args.sample_mode,
    )
    arch_min_trajectories, arch_results = compute_arch_instruction_results(
        experiment_rows,
        model_dates,
        turn_limits,
        args.sample_mode,
    )

    out_dir = args.out_dir or (args.experiments_csv.resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "correctness_rate_by_release_date.csv"
    speedup_summary_path = out_dir / "best_speedup_by_release_date.csv"
    arch_summary_path = out_dir / "arch_instruction_correctness_by_release_date.csv"

    write_summary_csv(results, turn_limits, summary_path)
    write_best_speedup_csv(speedup_results, turn_limits, speedup_summary_path)
    write_arch_instruction_csv(arch_results, turn_limits, arch_summary_path)

    if args.sample_mode == "all":
        print("Using all trajectories from each experiment")
    else:
        print(f"Using first {min_trajectories} trajectories from each experiment")
    print(f"Wrote {summary_path}")
    print(f"Wrote {speedup_summary_path}")
    print(f"Wrote {arch_summary_path}")
    if args.csv_only:
        print("Skipped grouped PNG generation (--csv-only)")
        return

    ptx_dates = load_release_dates(args.ptx_release_dates, "arch")
    fa_dates = load_fa_release_dates(args.fa_release_dates)
    grouped_plot_dir = out_dir / "by_workload_arch"
    grouped_plot_dir.mkdir(parents=True, exist_ok=True)

    grouped_plot_paths = []
    grouped_results = grouped_by_workload_arch(results)
    grouped_speedup_results = grouped_by_workload_arch(speedup_results)
    grouped_arch_results = grouped_by_workload_arch(arch_results)
    for workload_name, arch in sorted(grouped_results):
        workload_slug = slugify(workload_name)
        arch_slug = slugify(arch)
        group_rows = grouped_results[(workload_name, arch)]
        definition = group_rows[0].get("definition", "")
        title_context = workload_arch_title(group_rows[0])
        reference_lines = [
            ptx_reference_line(arch, ptx_dates),
            *fa_reference_lines(definition, arch, fa_dates),
        ]
        group_suffix = f"{workload_slug}__{arch_slug}.png"

        group_plot_path = grouped_plot_dir / f"correctness_rate_by_release_date__{group_suffix}"
        group_speedup_plot_path = grouped_plot_dir / f"best_speedup_by_release_date__{group_suffix}"
        group_arch_plot_path = grouped_plot_dir / f"arch_instruction_correctness_by_release_date__{group_suffix}"

        plot_results(
            grouped_results[(workload_name, arch)],
            min_trajectories,
            args.sample_mode,
            turn_limits,
            group_plot_path,
            reference_lines=reference_lines,
            title_context=title_context,
            model_colors=model_colors,
        )
        plot_best_speedup_by_release_date(
            grouped_speedup_results[(workload_name, arch)],
            speedup_min_trajectories,
            args.sample_mode,
            turn_limits,
            group_speedup_plot_path,
            reference_lines=reference_lines,
            title_context=title_context,
            model_colors=model_colors,
        )
        plot_arch_instruction_correctness_by_release_date(
            grouped_arch_results[(workload_name, arch)],
            arch_min_trajectories,
            args.sample_mode,
            turn_limits,
            group_arch_plot_path,
            reference_lines=reference_lines,
            title_context=title_context,
            model_colors=model_colors,
        )
        grouped_plot_paths.extend(
            [group_plot_path, group_speedup_plot_path, group_arch_plot_path]
        )

    print(f"Wrote {len(grouped_plot_paths)} grouped plots to {grouped_plot_dir}")


if __name__ == "__main__":
    main()
