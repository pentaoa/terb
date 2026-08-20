#!/usr/bin/env python3
"""Merge Terb streaming results with Beat This! truncated-audio baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


CHECKPOINT_DIRS = {"bpm_4s": "wav-4s", "bpm_8s": "wav-8s", "bpm_16s": "wav-16s", "bpm_30s": "wav"}


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rel(value: float, reference: float) -> float:
    return abs(value - reference) / max(reference, 1e-6)


def classify(value: float | None, reference: float) -> str:
    if value is None:
        return "no_result"
    if rel(value, reference) <= 0.04:
        return "correct"
    if rel(value, reference * 0.5) <= 0.04:
        return "half_time"
    if rel(value, reference * 2.0) <= 0.04:
        return "double_time"
    return "wrong"


def reference_rows(manifest: list[dict], root: Path) -> list[dict]:
    timings = {
        Path(item["input"]).stem: item["processing_time_secs"]
        for item in load(root / "wav" / "beat_this.json")["files"]
    }
    rows = []
    for item in manifest:
        stem = Path(item["path"]).stem
        checkpoints = {
            key: load(root / directory / f"{stem}.json").get("bpm")
            for key, directory in CHECKPOINT_DIRS.items()
        }
        final_bpm = checkpoints["bpm_30s"]
        rows.append(
            {
                "path": item["path"],
                "dataset": item["dataset"],
                "split": item["split"],
                "genre": item.get("genre", ""),
                "algorithm": "beat-this-small-reference",
                "reference_bpm": item["bpm"],
                **checkpoints,
                "final_bpm": final_bpm,
                "confidence": None,
                "first_estimate_seconds": None,
                "stable_lock_seconds": None,
                "locked_jitter_bpm": None,
                "classification": classify(final_bpm, item["bpm"]),
                "elapsed_seconds": timings[stem],
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    kinds = defaultdict(int)
    for row in rows:
        kinds[row["classification"]] += 1
    checkpoint_accuracy = {}
    for key in CHECKPOINT_DIRS:
        checkpoint_accuracy[key.replace("bpm_", "accuracy_")] = sum(
            value is not None and rel(value, row["reference_bpm"]) <= 0.04
            for row in rows
            if (value := row.get(key)) is not None
        ) / n
    locks = [row["stable_lock_seconds"] for row in rows if row.get("stable_lock_seconds") is not None]
    jitters = [row["locked_jitter_bpm"] for row in rows if row.get("locked_jitter_bpm") is not None]
    elapsed = sum(row.get("elapsed_seconds", 0.0) for row in rows)
    audio_seconds = 30.0 * n
    return {
        "count": n,
        "strict_accuracy": kinds["correct"] / n,
        "metrical_accuracy": (kinds["correct"] + kinds["half_time"] + kinds["double_time"]) / n,
        "half_time_rate": kinds["half_time"] / n,
        "double_time_rate": kinds["double_time"] / n,
        "wrong_rate": kinds["wrong"] / n,
        "unstable_rate": kinds["unstable"] / n,
        "no_result_rate": kinds["no_result"] / n,
        **checkpoint_accuracy,
        "mean_lock_seconds": mean(locks) if locks else None,
        "mean_jitter_bpm": mean(jitters) if jitters else None,
        "realtime_factor": elapsed / audio_seconds,
    }


def summaries(rows: list[dict]) -> dict:
    overall = {}
    by_genre = {}
    algorithms = sorted({row["algorithm"] for row in rows})
    genres = sorted({row["genre"] for row in rows})
    for algorithm in algorithms:
        overall[algorithm] = summarize([row for row in rows if row["algorithm"] == algorithm])
        by_genre[algorithm] = {
            genre: summarize([row for row in rows if row["algorithm"] == algorithm and row["genre"] == genre])
            for genre in genres
        }
    return {"tolerance": 0.04, "overall": overall, "by_genre": by_genre}


def report(summary: dict) -> str:
    lines = [
        "# GiantSteps fixed-subset three-way BPM benchmark",
        "",
        "20 MD5-verified Beatport previews, two tracks from each of ten most common genres; tolerance ±4%.",
        "Beat This! is offline/non-causal: its checkpoint columns are independent truncated-audio runs; lock/jitter are not applicable.",
        "",
        "|algorithm|n|strict|metrical|half|double|wrong|unstable|4s|8s|16s|30s|lock s|jitter|RTF|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["overall"].items():
        pct = lambda key: f'{100 * item[key]:.1f}%'
        number = lambda key: "–" if item[key] is None else f'{item[key]:.2f}'
        lines.append(
            f"|{name}|{item['count']}|{pct('strict_accuracy')}|{pct('metrical_accuracy')}|"
            f"{pct('half_time_rate')}|{pct('double_time_rate')}|{pct('wrong_rate')}|"
            f"{pct('unstable_rate')}|{pct('accuracy_4s')}|{pct('accuracy_8s')}|"
            f"{pct('accuracy_16s')}|{pct('accuracy_30s')}|{number('mean_lock_seconds')}|"
            f"{number('mean_jitter_bpm')}|{item['realtime_factor']:.3f}|"
        )
    lines.extend(["", "## Strict accuracy by genre", "", "|genre|" + "|".join(summary["overall"]) + "|", "|---|" + "---:|" * len(summary["overall"])])
    genres = sorted(next(iter(summary["by_genre"].values())))
    for genre in genres:
        values = [f"{100 * summary['by_genre'][name][genre]['strict_accuracy']:.0f}%" for name in summary["overall"]]
        lines.append(f"|{genre}|" + "|".join(values) + "|")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("terb_results", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    rows = load(args.terb_results) + reference_rows(manifest, args.reference_root)
    summary = summaries(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(report(summary), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
