"""
dcm_data_filter_parallel.py

[Phase 1] Series별 하위 폴더 분류  (dcm_data_filter_parallel)
  - {site}_axial 하위 환자 폴더 순회
  - 파일 10개 미만 → {site}_원본 으로 즉시 이동
  - 파일 10개 이상 → SeriesInstanceUID 기준 하위 폴더로 분류

[Phase 2] Contrast Series 선택  (select_contrast_series)
  - {site}_axial 하위 환자 폴더 순회
  - Series가 1개이면 SKIP
  - Series가 여러 개이면 우선순위에 따라 1개 선택,
    나머지는 {site}_axial_except 으로 이동 후 빈 폴더 삭제

사용법:
  python dcm_data_filter_parallel.py [--dry-run]
"""

import shutil
import argparse
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pydicom

SITES = ["강남", "신촌"]

_print_lock = threading.Lock()


def safe_print(msg: str):
    with _print_lock:
        print(msg)


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def read_dicom_tags(filepath: Path):
    try:
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)
        uid  = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN")).strip()
        desc = str(getattr(ds, "SeriesDescription", "")).strip()
        return uid, desc
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Series별 하위 폴더 분류
# ══════════════════════════════════════════════════════════════════════════════

def p1_find_dcm_files(patient_dir: Path) -> list:
    """DICOM 파일 목록만 빠르게 수집 (태그 읽기 없음)."""
    dcm_files = list(patient_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in patient_dir.iterdir()
                     if f.is_file() and not f.suffix]
    if not dcm_files:
        for subdir in patient_dir.iterdir():
            if not subdir.is_dir():
                continue
            sub_dcm = list(subdir.glob("*.dcm"))
            if not sub_dcm:
                sub_dcm = [f for f in subdir.iterdir()
                           if f.is_file() and not f.suffix]
            dcm_files.extend(sub_dcm)
    return dcm_files


def p1_collect_series(dcm_files: list) -> dict:
    series_map = defaultdict(lambda: {"desc": "", "files": []})
    for fp in dcm_files:
        uid, desc = read_dicom_tags(fp)
        if uid is None:
            continue
        series_map[uid]["files"].append(fp)
        if not series_map[uid]["desc"] and desc:
            series_map[uid]["desc"] = desc
    return dict(series_map)


def p1_make_folder_name(uid: str, desc: str) -> str:
    return sanitize(desc) if desc else f"series_{uid[:8]}"


def p1_move_files(files: list, target_dir: Path, uid_prefix: str = ""):
    target_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        dest = target_dir / fp.name
        if dest.exists():
            dest = target_dir / f"{fp.stem}_{uid_prefix or fp.parent.name}{fp.suffix}"
        shutil.move(str(fp), str(dest))


def p1_process_patient(patient_dir: Path, dst_root: Path, site: str,
                        dry_run: bool, log_lines: list):
    def log(msg):
        log_lines.append(msg)
        safe_print(f"[{site}][P1] {msg}")

    patient_name = patient_dir.name
    log(f"\n{'='*60}")
    log(f"[환자] {patient_name}")

    dcm_files = p1_find_dcm_files(patient_dir)
    if not dcm_files:
        log("  → DCM 파일 없음. SKIP.")
        return

    total = len(dcm_files)

    if total < 10:
        log(f"  → 파일 {total}개 (10개 미만) → {site}_원본/{patient_name}/ [소량 즉시 이동]")
        if not dry_run:
            p1_move_files(dcm_files, dst_root / patient_name)
        return

    series_map = p1_collect_series(dcm_files)
    if not series_map:
        log("  → DICOM 태그 읽기 실패. SKIP.")
        return

    log(f"  → 파일 {total}개, Series {len(series_map)}개 발견:")
    for uid, info in series_map.items():
        desc       = info["desc"]
        files      = info["files"]
        file_count = len(files)
        folder_name = p1_make_folder_name(uid, desc)

        if file_count < 10:
            log(f"     └─ '{desc or uid[:20]}'  파일 {file_count}개"
                f"  →  {site}_원본/{patient_name}/  [소량 이동]")
            if not dry_run:
                p1_move_files(files, dst_root / patient_name, uid[:6])
        else:
            target_dir = patient_dir / folder_name
            log(f"     └─ '{desc or uid[:20]}'  {file_count}개 파일"
                f"  →  {patient_name}/{folder_name}/")
            if not dry_run:
                p1_move_files(files, target_dir, uid[:6])

    log(f"  총 {total}개 파일 처리 완료.")




# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Contrast Series 선택
# ══════════════════════════════════════════════════════════════════════════════

PRIORITY_RULES = [
    (50, r"portal|venous|문맥|정맥기|pv\b|pvp|port"),
    (45, r"arterial|동맥기|art\b|hap|ap\b"),
    (40, r"delay|late|지연|equilib|with"),
    (30, r"contrast|contras|\+c\b|ce\b|c\+|조영|enhanced|enhance|wash"),
    (20, r"post.*contrast|post.*ce"),
    (5,  r"non.?contrast|without|plain|native|단순|비조영|nc\b|unenhanced|pre\b|precontrast"),
]


def p2_get_priority(series_desc: str) -> int:
    desc = (series_desc or "").lower()
    best = 0
    for score, pattern in PRIORITY_RULES:
        if re.search(pattern, desc, re.IGNORECASE):
            best = max(best, score)
    return best


def p2_collect_series(patient_dir: Path) -> dict:
    series_map = defaultdict(lambda: {"desc": "", "files": []})
    dcm_files = list(patient_dir.rglob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in patient_dir.rglob("*")
                     if f.is_file() and not f.suffix]
    for fp in dcm_files:
        uid, desc = read_dicom_tags(fp)
        if uid is None:
            continue
        series_map[uid]["files"].append(fp)
        if not series_map[uid]["desc"] and desc:
            series_map[uid]["desc"] = desc
    return dict(series_map)


def p2_choose_best_series(series_map: dict) -> str:
    ranked = sorted(
        series_map.items(),
        key=lambda item: (p2_get_priority(item[1]["desc"]), len(item[1]["files"])),
        reverse=True,
    )
    return ranked[0][0]


def p2_move_files(files: list, dst_base: Path, patient_dir: Path, dry_run: bool) -> int:
    moved = 0
    for fp in files:
        rel    = fp.relative_to(patient_dir)
        target = dst_base / patient_dir.name / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            shutil.move(str(fp), str(target))
        moved += 1
    return moved


def p2_process_patient(patient_dir: Path, dst_root: Path, site: str,
                        dry_run: bool, log_lines: list):
    def log(msg):
        log_lines.append(msg)
        safe_print(f"[{site}][P2] {msg}")

    patient_name = patient_dir.name
    log(f"\n{'='*60}")
    log(f"[환자] {patient_name}")

    series_map = p2_collect_series(patient_dir)
    if not series_map:
        log("  → DCM 파일 없음. SKIP.")
        return

    if len(series_map) == 1:
        uid, info = next(iter(series_map.items()))
        log(f"  → Series 1개 ({info['desc'] or uid}). SKIP.")
        return

    log(f"  → Series {len(series_map)}개 발견:")
    for uid, info in series_map.items():
        score = p2_get_priority(info["desc"])
        log(f"     UID={uid[:20]}...  desc='{info['desc']}'  "
            f"슬라이스={len(info['files'])}  점수={score}")

    best_uid  = p2_choose_best_series(series_map)
    best_desc = series_map[best_uid]["desc"]
    log(f"  ★ 선택: '{best_desc}' (UID={best_uid[:20]}...)")

    total_moved = 0
    for uid, info in series_map.items():
        if uid == best_uid:
            continue
        n      = p2_move_files(info["files"], dst_root, patient_dir, dry_run)
        action = "이동 예정" if dry_run else "이동 완료"
        log(f"  → '{info['desc']}' Series {n}개 파일 {action} → {site}_axial_except/{patient_name}/")
        total_moved += n

    log(f"  총 {total_moved}개 파일 처리 완료.")

    # 빈 하위 폴더 삭제
    if not dry_run:
        for subdir in sorted(patient_dir.rglob("*"), reverse=True):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
                log(f"  → 빈 폴더 삭제: {subdir}")


# ══════════════════════════════════════════════════════════════════════════════
# 통합 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_site(site: str, dry_run: bool):
    """환자별로 Phase 1 → Phase 2 순서로 처리."""
    src_root  = Path(rf"D:\데이터서비스팀 영상제공\{site}\{site}_axial")
    dst_root1 = Path(rf"D:\데이터서비스팀 영상제공\{site}\{site}_원본")
    dst_root2 = Path(rf"D:\데이터서비스팀 영상제공\{site}\{site}_axial_except")
    log_lines = []

    safe_print(f"\n[{site}] 시작  SRC={src_root}")

    if not src_root.exists():
        safe_print(f"[{site}] [ERROR] 경로 없음: {src_root}")
        return

    dst_root1.mkdir(parents=True, exist_ok=True)
    dst_root2.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    safe_print(f"[{site}] 총 환자 폴더 수: {len(patient_dirs)}")
    log_lines.append(f"총 환자 폴더 수: {len(patient_dirs)}")

    for pdir in patient_dirs:
        try:
            p1_process_patient(pdir, dst_root1, site, dry_run, log_lines)
        except Exception as e:
            msg = f"  [SKIP][P1] {pdir.name} 처리 중 오류: {e}"
            log_lines.append(msg)
            safe_print(f"[{site}] {msg}")

        try:
            p2_process_patient(pdir, dst_root2, site, dry_run, log_lines)
        except Exception as e:
            msg = f"  [SKIP][P2] {pdir.name} 처리 중 오류: {e}"
            log_lines.append(msg)
            safe_print(f"[{site}] {msg}")

    log_path = src_root.parent / "pipeline_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    safe_print(f"[{site}] 완료  로그: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="[Phase1] DCM Series 분류 → [Phase2] Contrast Series 선택 (병렬)"
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 이동 없이 결과 출력")
    args = parser.parse_args()

    if args.dry_run:
        print("※ DRY-RUN 모드: 실제 파일 이동을 수행하지 않습니다.\n")

    print(f"병렬 처리 SITE: {SITES}\n")

    with ThreadPoolExecutor(max_workers=len(SITES)) as executor:
        futures = {executor.submit(run_site, site, args.dry_run): site
                   for site in SITES}
        for future in as_completed(futures):
            site = futures[future]
            try:
                future.result()
            except Exception as e:
                safe_print(f"[{site}] 예외 발생: {e}")

    print("\n[전체 완료]")


if __name__ == "__main__":
    main()
