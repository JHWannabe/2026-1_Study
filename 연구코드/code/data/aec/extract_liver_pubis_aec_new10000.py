"""
[목적] 새 데이터 10000명 코호트 전체(new10000_final_dataset.xlsx의 metadata에 있는 10578명)에
대해 extract_liver_pubis_aec_new10000_sample.py로 검증한 파이프라인(liver/hip 랜드마크 →
z_bounds → crop → tissue_4_types → VAT/SAT/TAMA/LAMA/NAMA/IMATA)을 실행해 저장한다.

[aec_cropped/aec_128] TotalSegmentator liver/hip 기반으로 찾은 "간 상단~두덩뼈 하단" 크롭
구간을 process_patient_new10000()이 체성분과 동일한 lo/hi 범위로 함께 계산한
aec_cropped_liver_to_pubis를 이 스크립트가 저장해, body_comp_cropped와 슬라이스 수·경계가
항상 일치하도록 한다.

[aec_total] gangnam_final_dataset.xlsx와 동일한 구조를 맞추기 위해, 크롭 전 series 전체의
AEC(XRayTubeCurrent) 곡선도 세그멘테이션 성공 여부와 무관하게 함께 저장한다
(process_patient_new10000()의 aec_total_row → base._write_aec_total() 재사용).

[재시작 가능]  BATCH_SIZE명마다 체크포인트(.pkl)와 출력 xlsx를 저장한다. 중간에 중단해도
다시 실행하면 이미 처리된 PatientID는 건너뛰고 이어서 진행한다(기존 결과 보존).
출력 파일들(z_bounds/body_comp/body_comp_cropped/aec_cropped/aec_total)의 append/dedup
로직은 gangnam/신촌에 쓰는 extract_liver_pubis_aec.py의 _write_zbounds/
_write_body_composition/_write_body_comp_cropped/_write_aec_cropped/_write_aec_total을
그대로 재사용한다.

[GPU 병렬 처리] TotalSegmentator 호출은 기본값(nr_thr_saving=6)으로 저장용
서브프로세스를 내부에서 또 띄우는데, 외부 워커와 같이 쓰면 그만큼 프로세스 수가
곱해져 CPU 8코어를 두고 경쟁할 수 있다. 그래서 extract_liver_pubis_aec_new10000_sample.py의
totalsegmentator 호출에 nr_thr_resamp=1/nr_thr_saving=1을 고정해 그 내부 서브프로세스를
없애고, 워커 수(N_WORKERS)와 워커당 스레드 수(OMP_NUM_THREADS)를 물리 코어 8개에
맞춰 나눠 CPU 과다구독을 막는다. 다만 Windows는 NVIDIA MPS(여러 프로세스가 GPU
컨텍스트를 동시에 쓰는 기능, Linux 전용)가 없어 GPU 연산 자체는 어차피 순서대로
처리된다 — 대신 전처리(CPU)와 다른 환자의 GPU 연산이 겹칠 여지로 이득을 본다.
# ponytail: N_WORKERS/THREADS_PER_WORKER는 이 컴퓨터(8코어/RTX 5090) 기준 시작값.
# 작업관리자·nvidia-smi로 보고 조정.

[진행 상황] tqdm 진행률 표시줄로 콘솔에서 실시간 확인 가능.

[4단계 자동 실행] STEP4_INTERVAL명 처리할 때마다 9_merge_body_composition.py를 그대로
불러와 실행해, new10000_final_dataset.xlsx가 3단계 중간 배치를 계속 반영하도록 한다
(매 체크포인트(BATCH_SIZE=5)마다 돌리기엔 4단계 자체가 큰 xlsx를 여러 개 다시 읽고 써서
비용이 크므로 더 큰 주기로 실행). 루프 종료 시(전체 완료) 한 번 더 실행해 마지막 배치도
반영한다.
"""

import concurrent.futures as cf
import importlib.util
import os
import pickle
import sys
import tempfile
from pathlib import Path

N_WORKERS = 2
THREADS_PER_WORKER = 4  # N_WORKERS * THREADS_PER_WORKER ≈ 물리 코어 수(8). 4는 Windows에서
# 워커 스폰 시 PermissionError(WinError 5, DuplicateHandle 거부)가 나 되돌림.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_liver_pubis_aec as base  # noqa: E402
from extract_liver_pubis_aec_new10000_sample import (  # noqa: E402
    find_patient_dir, process_patient_new10000,
)

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "새 데이터 10000명"
FINAL_DATASET_PATH = DATA_DIR / "new10000_final_dataset.xlsx"
BEST_SERIES_PATH = DATA_DIR / "best_series.xlsx"
CHECKPOINT_PATH = DATA_DIR / "new10000_bodycomp_checkpoint.pkl"

PATHS = {
    "_site": "새데이터10000",
    "z_bounds": str(DATA_DIR / "new10000_z_bounds.xlsx"),
    # body_comp/body_comp_cropped, aec_cropped/aec_total은 각각 같은 파일에 시트로 합쳐 저장한다.
    "body_comp": str(DATA_DIR / "new10000_body_composition.xlsx"),
    "body_comp_cropped": str(DATA_DIR / "new10000_body_composition.xlsx"),
    "aec_cropped": str(DATA_DIR / "new10000_aec_cropped.xlsx"),
    "aec_total": str(DATA_DIR / "new10000_aec_cropped.xlsx"),
}

BATCH_SIZE = 5
STEP4_INTERVAL = 50  # ponytail: 이만큼 새로 처리될 때마다 4단계(병합) 실행. 너무 잦으면
# 4단계 자체의 큰 xlsx 재작성 비용이 커지고, 너무 뜸하면 final_dataset.xlsx가 오래 뒤처진다.


def _run_merge_step() -> None:
    """9_merge_body_composition.py(4단계)를 그대로 불러와 실행 — 지금까지의 3단계 산출물로
    new10000_final_dataset.xlsx를 갱신한다."""
    merge_path = Path(__file__).resolve().parent.parent / "9_merge_body_composition.py"
    spec = importlib.util.spec_from_file_location("merge_step", merge_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _worker_init() -> None:
    """워커 프로세스 시작 시 1회 실행 — nnU-Net/torch가 코어를 전부 잡지 않도록 워커당
    스레드 수를 제한한다(N_WORKERS개 워커가 동시에 물리 코어를 다 쓰려 들면 오히려
    스레드 경합으로 느려진다)."""
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)


def _run_one(job: tuple) -> tuple:
    """워커 프로세스에서 실행 — 계산만 하고 파일 쓰기는 메인 프로세스에서 순차 처리한다
    (여러 프로세스가 같은 xlsx를 동시에 덮어쓰면 깨지므로)."""
    pid, patient_dir, series_desc, kvp, tmp_dir = job
    if patient_dir is None:
        return pid, {"seg_status": "no_dicom"}, kvp
    try:
        result = process_patient_new10000(pid, patient_dir, series_desc, tmp_dir)
    except Exception as e:
        result = {"PatientID": pid, "seg_status": f"error:{type(e).__name__}:{e}"}
    return pid, result, kvp


def load_targets() -> pd.DataFrame:
    """new10000_final_dataset.xlsx의 metadata에 있는 환자(10578명) x best_series의 선정 series."""
    meta_ids = pd.read_excel(FINAL_DATASET_PATH, sheet_name="metadata", usecols=["PatientID"])
    best = pd.read_excel(BEST_SERIES_PATH, usecols=["PatientID", "series_desc", "kVp"])
    return best.merge(meta_ids, on="PatientID")


def load_checkpoint() -> set[int]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            return pickle.load(f)
    return set()


def save_checkpoint(done: set[int]) -> None:
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(done, f)


def main():
    targets = load_targets()
    done = load_checkpoint()
    if PATHS.get("_z_bounds_exists") is None and Path(PATHS["z_bounds"]).exists():
        done |= set(pd.read_excel(PATHS["z_bounds"])["PatientID"].astype(int))
    pending = targets[~targets["PatientID"].isin(done)]
    print(f"대상 {len(targets)}명 | 완료 {len(done)}명 | 처리 예정 {len(pending)}명 | "
          f"워커 {N_WORKERS}개 x 스레드 {THREADS_PER_WORKER}개")

    print("  4단계(병합) 실행 중... (재시작 시 1회, 3단계 시작 전)")
    _run_merge_step()

    z_rows, bc_rows, aec_entries, aec_total_rows = [], [], [], []
    n_processed = 0
    last_step4_done = len(done)  # done이 STEP4_INTERVAL의 배수가 아닌 값(예: 백업 병합)에서
    # 시작해도 다음 문턱을 정확히 넘을 때 실행되도록 "마지막 실행 시점 대비 증가량"으로 판단한다
    # (len(done) % STEP4_INTERVAL == 0은 시작 offset이 안 맞으면 영원히 안 걸릴 수 있음).
    with tempfile.TemporaryDirectory() as tmp_dir:
        jobs = [(int(r["PatientID"]), find_patient_dir(int(r["PatientID"])), r["series_desc"], r["kVp"], tmp_dir)
                for _, r in pending.iloc[::-1].iterrows()]  # PatientID 역순으로 처리

        with cf.ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
            for pid, result, kvp in tqdm(ex.map(_run_one, jobs, chunksize=1),
                                          total=len(jobs), desc="체성분 추출 (병렬)"):
                z_rows.append({
                    "PatientID": pid,
                    "series_description": result.get("series_description"),
                    "manufacturer_model": result.get("manufacturer_model"),
                    "n_slices": result.get("n_slices"),
                    "inst_min": result.get("inst_min"),
                    "slices": result.get("slices"),
                    "liver_upper_slice": result.get("liver_upper_slice"),
                    "pubis_slice": result.get("pubis_slice"),
                    "liver_upper_k": result.get("liver_upper_k"),
                    "pubis_k": result.get("pubis_k"),
                    "z_range": result.get("z_range"),
                    "z_liver_upper": result.get("z_liver_upper"),
                    "z_pubis": result.get("z_pubis"),
                    "seg_status": result["seg_status"],
                })
                bc = result.get("body_composition")
                if bc is not None:
                    bc_rows.append({"PatientID": pid, "series_description": result.get("series_description"),
                                    "manufacturer_model": result.get("manufacturer_model"), **bc, "kVp": kvp})

                aec_cropped = result.get("aec_cropped_liver_to_pubis")
                if aec_cropped:
                    # kVp는 best_series.xlsx에서 선정 시 이미 읽어둔 값 — 체성분과 동일한
                    # lo/hi 크롭이므로 slice 수는 body_comp_cropped와 항상 일치한다.
                    aec_entries.append({
                        "PatientID": pid, "series_description": result.get("series_description"),
                        "manufacturer_model": result.get("manufacturer_model"), "kVp": kvp,
                        "z_range": result.get("z_range"), "aec_cropped": aec_cropped,
                    })

                aec_total_row = result.get("aec_total_row")
                if aec_total_row is not None:
                    aec_total_rows.append(aec_total_row)

                done.add(pid)
                n_processed += 1
                if n_processed % BATCH_SIZE == 0 or n_processed == len(jobs):
                    base._write_zbounds(PATHS, z_rows)
                    base._write_body_composition(PATHS, bc_rows)
                    base._write_body_comp_cropped(PATHS, bc_rows)
                    base._write_aec_cropped(PATHS, aec_entries)
                    base._write_aec_total(PATHS, aec_total_rows)
                    save_checkpoint(done)
                    z_rows, bc_rows, aec_entries, aec_total_rows = [], [], [], []
                    tqdm.write(f"  체크포인트 저장 | {n_processed}/{len(jobs)}")

                    if len(done) - last_step4_done >= STEP4_INTERVAL or n_processed == len(jobs):
                        tqdm.write("  4단계(병합) 실행 중...")
                        _run_merge_step()
                        last_step4_done = len(done)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print("전체 코호트 체성분 추출 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        crash_log = DATA_DIR / "new10000_pipeline_crash.log"
        with open(crash_log, "a", encoding="utf-8") as f:
            import datetime
            f.write(f"\n{'='*70}\n{datetime.datetime.now()}\n")
            traceback.print_exc(file=f)
        raise
