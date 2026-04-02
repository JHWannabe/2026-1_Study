from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SITES = ["강남", "신촌"]
SCRIPT_ORDER = [
    "0_image_filter.py",
    "1_aec_group_comparison.py",
    "2_bmi_tama.py",
    "3_aec_cluster_analysis.py",
    "3_aec_cluster_analysis copy.py",
    "3_aec_quadrant_shape_analysis.py",
    "4_aec_k_means.py",
]

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parents[1]
OUTPUT_DIR = BASE_DIR / "result/센터별"


@dataclass
class RunResult:
    site_name: str
    script_name: str
    return_code: int


# CLI and execution helpers
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 0323 analysis scripts for 강남 and 신촌."
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=SITES,
        help="Site names to run. Default: 강남 신촌",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any script fails.",
    )
    return parser.parse_args()


def build_env(site_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["AEC_SITE_NAME"] = site_name
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_script(script_path: Path, site_name: str, output_dir: Path) -> RunResult:
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(CODE_DIR),
        env=build_env(site_name),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if completed.stdout.strip():
        print(completed.stdout.rstrip())
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)

    return RunResult(
        site_name=site_name,
        script_name=script_path.name,
        return_code=completed.returncode,
    )


# Entry point
def main() -> None:
    args = parse_args()
    script_paths = [CODE_DIR / script_name for script_name in SCRIPT_ORDER]
    missing_scripts = [path.name for path in script_paths if not path.is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing scripts: {missing_scripts}")

    results: list[RunResult] = []
    for site_name in args.sites:
        site_output_dir = OUTPUT_DIR / site_name
        for script_path in script_paths:
            print(f"[Run] site={site_name} script={script_path.name}")
            result = run_script(script_path, site_name, site_output_dir)
            results.append(result)
            if result.return_code == 0:
                print(f"[OK] {site_name} | {script_path.name}")
            else:
                print(f"[FAIL] {site_name} | {script_path.name}")

    failed_count = sum(result.return_code != 0 for result in results)
    print(f"Completed runs: {len(results)}")
    print(f"Failures: {failed_count}")
    print(f"Summary saved to: {OUTPUT_DIR}")
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
