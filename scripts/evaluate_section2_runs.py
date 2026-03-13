#!/usr/bin/env python3
import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_decoder.lm_utils import wer
from neural_decoder.section2_utils import read_jsonl


def parse_metrics_file(path):
    metrics = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            try:
                metrics[key] = ast.literal_eval(value)
            except Exception:
                metrics[key] = value
    return metrics


def utterance_word_error(reference, hypothesis):
    return wer(reference.split(), hypothesis.split())


def analyze_run(run_name, run_dir):
    metrics_path = run_dir / "test_metrics.txt"
    analysis_path = run_dir / "test_analysis.jsonl"
    metrics = parse_metrics_file(metrics_path)
    records = read_jsonl(analysis_path)

    trigger_count = 0
    oracle_gap_numerator = 0
    oracle_gap_denominator = 0
    close_errors = []
    distant_errors = []
    high_confusion_errors = []
    low_confusion_errors = []
    pair_stats = defaultdict(lambda: {"count": 0, "correct": 0})

    for record in records:
        reference = record["reference"]
        selected = record["selected_text"]
        selected_err = utterance_word_error(reference, selected)

        baseline = record["ranked_candidates"][0]["text"] if record["ranked_candidates"] else ""
        baseline_err = utterance_word_error(reference, baseline)

        candidate_pool = [candidate["text"] for candidate in record.get("ranked_candidates", [])]
        candidate_pool.extend(record.get("expanded_candidates", []))
        if candidate_pool:
            oracle_err = min(utterance_word_error(reference, candidate) for candidate in candidate_pool)
            oracle_gap_denominator += max(baseline_err - oracle_err, 0)
            oracle_gap_numerator += max(baseline_err - selected_err, 0)

        if record.get("triggered", False):
            trigger_count += 1
            high_confusion_errors.append(selected_err)
        else:
            low_confusion_errors.append(selected_err)

        confusion_spans = record.get("confusion_spans", [])
        if confusion_spans:
            min_distance = min(span.get("min_phoneme_distance", 0) for span in confusion_spans)
            if min_distance <= 3:
                close_errors.append(selected_err)
            else:
                distant_errors.append(selected_err)

            for span in confusion_spans:
                alternatives = tuple(
                    sorted(
                        alt["text"]
                        for alt in span.get("alternatives", [])
                        if alt.get("text")
                    )
                )
                if len(alternatives) < 2:
                    continue
                stats = pair_stats[alternatives]
                stats["count"] += 1
                stats["correct"] += int(selected_err == 0)

    pair_summary = []
    for alternatives, stats in sorted(
        pair_stats.items(), key=lambda item: item[1]["count"], reverse=True
    )[:20]:
        pair_summary.append(
            {
                "alternatives": list(alternatives),
                "count": stats["count"],
                "accuracy": stats["correct"] / max(stats["count"], 1),
            }
        )

    return {
        "run_name": run_name,
        "metrics": metrics,
        "trigger_rate": trigger_count / max(len(records), 1),
        "oracle_gap_closed": oracle_gap_numerator / max(oracle_gap_denominator, 1),
        "high_confusion_avg_word_errors": (
            sum(high_confusion_errors) / len(high_confusion_errors)
            if high_confusion_errors
            else 0.0
        ),
        "low_confusion_avg_word_errors": (
            sum(low_confusion_errors) / len(low_confusion_errors) if low_confusion_errors else 0.0
        ),
        "close_confusion_avg_word_errors": (
            sum(close_errors) / len(close_errors) if close_errors else 0.0
        ),
        "distant_confusion_avg_word_errors": (
            sum(distant_errors) / len(distant_errors) if distant_errors else 0.0
        ),
        "top_confusion_pairs": pair_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the Section 2 ablation runs into a report."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        required=True,
        help="Named run directory containing test_metrics.txt and test_analysis.jsonl.",
    )
    parser.add_argument(
        "--outputPath",
        type=Path,
        default=Path("derived/section2_report.json"),
        help="Where to write the aggregated report JSON.",
    )
    args = parser.parse_args()

    report = []
    for name, directory in args.run:
        report.append(analyze_run(name, Path(directory).expanduser().resolve()))

    output_path = args.outputPath.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote Section 2 evaluation report to {output_path}")


if __name__ == "__main__":
    main()
