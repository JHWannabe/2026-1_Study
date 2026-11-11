@echo off
chcp 65001 >nul
cd /d "%~dp0"
python partial_tissue_images.py 강남 신촌
pause
