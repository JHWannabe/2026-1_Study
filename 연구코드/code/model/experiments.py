"""
단계별 성능 비교 실험 실행기.

환경변수 EXPERIMENT_STAGE를 통해 config.py의 하이퍼파라미터를 동적으로 적용하고
각 스테이지의 main.py를 순차 실행한다. 모든 스테이지 완료 후 AUC 비교 테이블을 출력한다.

사용법:
  python 연구코드/code/model/experiments.py              # 전 스테이지(0~6) 순차 실행
  python 연구코드/code/model/experiments.py --stages 0,1,2   # 특정 스테이지만
  python 연구코드/code/model/experiments.py --from-stage 3   # 3번부터 끝까지
  python 연구코드/code/model/experiments.py --compare-only   # 실행 없이 결과 비교만
"""

import os
import sys
import re
import argparse
import subprocess

# experiments.py 위치에서 config를 import할 수 있도록 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXPERIMENT_STAGES, EPOCHS  # noqa: E402

MAIN_PY      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


# ── 스테이지 실행 ─────────────────────────────────────────────

def run_stage(stage_id: int) -> int:
    """주어진 스테이지를 서브프로세스로 실행. 종료 코드를 반환."""
    cfg = EXPERIMENT_STAGES[stage_id]
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Stage {stage_id}: {cfg['desc']}")
    print(f"  LR={cfg['LR_RATE']}  HIDDEN={cfg['HIDDEN']}  N_HEADS={cfg['N_HEADS']}"
          f"  N_BLOCKS={cfg['N_BLOCKS']}  GRAD_CLIP={cfg['GRAD_CLIP']}  N_CA_LAYERS={cfg['N_CA_LAYERS']}")
    print(f"{sep}\n")

    env = os.environ.copy()
    env["EXPERIMENT_STAGE"] = str(stage_id)

    result = subprocess.run(
        [sys.executable, MAIN_PY],
        env=env,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"\n  [Stage {stage_id}] FAILED (returncode={result.returncode})")
    else:
        print(f"\n  [Stage {stage_id}] COMPLETED")

    return result.returncode


# ── 결과 파싱 ─────────────────────────────────────────────────

def _results_dir(stage_id: int) -> str:
    """스테이지 결과 디렉토리 경로를 반환."""
    cfg = EXPERIMENT_STAGES[stage_id]
    lr  = cfg["LR_RATE"]
    return os.path.join(PROJECT_ROOT, f"연구코드/results/0605/stage{stage_id}_{lr}_epoch{EPOCHS}")


def _parse_best_auc(md_path: str) -> dict[str, float]:
    """scaling_comparison_*.md 파일에서 Best Cases Summary 테이블을 파싱.

    반환 형식: {"M1": 0.803, "M2": 0.XXX, "M2_2": 0.XXX, "M3": 0.XXX}
    """
    if not os.path.exists(md_path):
        return {}
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    result: dict[str, float] = {}
    in_summary = False
    for line in content.splitlines():
        if "Best Cases Summary" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("---"):
            break
        if not in_summary:
            continue
        # 테이블 행 형식: | M1 | LR | scale_all | 0.8030 | ...
        m = re.match(r"\|\s*(M\d+(?:_\d+)?)\s*\|[^|]+\|[^|]+\|\s*([\d.]+)", line)
        if m:
            result[m.group(1)] = float(m.group(2))
    return result


def compare_stages(stages: list[int], mode: str = "crop80") -> None:
    """완료된 스테이지들의 Best AUC를 비교 테이블로 출력."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Stage 비교 — Best Test AUC  (mode={mode})")
    print(sep)
    print(f"  {'Stage':<8} {'설명':<35} {'M1':>7} {'M2':>7} {'M2_2':>7} {'M3':>7}")
    print("  " + "-" * 66)

    for sid in stages:
        cfg     = EXPERIMENT_STAGES.get(sid, {})
        desc    = cfg.get("desc", "")
        rdir    = _results_dir(sid)
        md_path = os.path.join(rdir, "comparison", mode, f"scaling_comparison_{mode}.md")
        aucs    = _parse_best_auc(md_path)

        def _fmt(key):
            v = aucs.get(key)
            return f"{v:.4f}" if v is not None else "  N/A "

        status = "" if aucs else "  (미완료)"
        print(f"  {sid:<8} {desc:<35} {_fmt('M1'):>7} {_fmt('M2'):>7} {_fmt('M2_2'):>7} {_fmt('M3'):>7}{status}")

    print(sep)


# ── 메인 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="단계별 하이퍼파라미터 실험 실행기",
    )
    parser.add_argument(
        "--stages", type=str, default=None,
        help="콤마 구분 스테이지 번호 (예: 0,1,2). 미입력시 전체 실행",
    )
    parser.add_argument(
        "--from-stage", type=int, default=None,
        help="해당 스테이지부터 마지막까지 실행",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="실행 없이 기존 결과 비교만 출력",
    )
    args = parser.parse_args()

    all_stage_ids = sorted(EXPERIMENT_STAGES.keys())

    if args.stages:
        target = [int(s.strip()) for s in args.stages.split(",")]
    elif args.from_stage is not None:
        target = [s for s in all_stage_ids if s >= args.from_stage]
    else:
        target = all_stage_ids

    # 계획 출력
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Experiment Stages 계획")
    print(sep)
    for sid in all_stage_ids:
        cfg    = EXPERIMENT_STAGES[sid]
        marker = " ◀ 실행예정" if (sid in target and not args.compare_only) else ""
        print(f"  Stage {sid}: {cfg['desc']}{marker}")
    print()

    if not args.compare_only:
        failed = []
        for sid in target:
            if sid not in EXPERIMENT_STAGES:
                print(f"  [Stage {sid}] 정의 없음, 스킵")
                continue
            rc = run_stage(sid)
            if rc != 0:
                failed.append(sid)

        if failed:
            print(f"\n  경고: 다음 스테이지 실패 → {failed}")

    # 결과 비교
    compare_stages(all_stage_ids, mode="crop80")
    compare_stages(all_stage_ids, mode="raw128")


if __name__ == "__main__":
    main()
