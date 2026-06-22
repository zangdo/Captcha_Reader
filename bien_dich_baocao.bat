@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Bien dich BAO CAO LaTeX - baocao.tex
echo ========================================

where pdflatex >nul 2>&1
if errorlevel 1 (
    echo [LOI] Chua cai MiKTeX hoac TeX Live.
    echo Tai MiKTeX: https://miktex.org/download
    echo Hoac dung Overleaf: upload baocao.tex + figures/
    pause
    exit /b 1
)

echo Lan 1...
pdflatex -interaction=nonstopmode baocao.tex
echo Lan 2...
pdflatex -interaction=nonstopmode baocao.tex
echo Lan 3 (cap nhat muc luc)...
pdflatex -interaction=nonstopmode baocao.tex

if exist baocao.pdf (
    echo.
    echo [OK] Da tao: baocao.pdf
    start "" baocao.pdf
) else (
    echo [LOI] Khong tao duoc PDF. Kiem tra log: baocao.log
)

pause
