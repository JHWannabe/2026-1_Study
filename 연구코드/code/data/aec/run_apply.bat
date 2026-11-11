@echo off
chcp 65001 >nul
echo 검토 표(series_reselect_review.xlsx) 생성 대기 중...
:w
if exist ..\..\series_reselect_review.xlsx goto go
timeout /t 10 /nobreak >nul
goto w
:go
python -X utf8 apply_reselect.py
echo.
echo 종료. 아무 키나 누르면 닫힙니다.
pause >nul
