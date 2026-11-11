@echo off
chcp 65001 >nul
echo 신촌 재처리(PID 20816) 종료 대기 중...
:w
tasklist /FI "PID eq 20816" 2>nul | find "20816" >nul
if not errorlevel 1 (
  timeout /t 20 /nobreak >nul
  goto w
)
echo 신촌 재처리 종료. 이후 작업 시작
python -X utf8 backfill_thickness.py 강남 신촌
python -X utf8 refresh_meta.py 강남
python -X utf8 refresh_meta.py 신촌
echo.
echo 모두 완료. 아무 키나 누르면 닫힙니다.
pause >nul
