@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo  Bien dich slide LaTeX - presentation.tex
echo ========================================

where pdflatex >nul 2>&1
if errorlevel 1 (
    echo.
    echo [LOI] Chua cai MiKTeX hoac TeX Live.
    echo.
    echo Cach 1 - Cai MiKTeX: https://miktex.org/download
    echo Cach 2 - Dung Overleaf: upload presentation.tex + figures/
    echo.
    pause
    exit /b 1
)

echo Dang bien dich lan 1...
pdflatex -interaction=nonstopmode presentation.tex
echo Dang bien dich lan 2...
pdflatex -interaction=nonstopmode presentation.tex

if not exist presentation.pdf (
    echo.
    echo [LOI] Khong tao duoc presentation.pdf - xem loi o tren.
    pause
    exit /b 1
)

echo.
echo [OK] Da tao: presentation.pdf
echo.
echo Mo PDF de trinh chieu...
start "" presentation.pdf

echo.
echo ========================================
echo  CACH TRINH CHIEU SLIDE
echo ========================================
echo.
echo  SumatraPDF (neu dung): nhan F5 = toan man hinh
echo  Microsoft Edge / Chrome: mo PDF, nhan F11 = toan man hinh
echo  Adobe Reader: Ctrl+L = toan man hinh
echo.
echo  Dieu huong:
echo    - Phim MUI TEN TRAI / PHAI  : slide truoc / sau
echo    - Phim HOME / END           : slide dau / cuoi
echo    - ESC                       : thoat toan man hinh
echo.
pause
