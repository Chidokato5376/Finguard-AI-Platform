@echo off
rem Chi chay bang xep hang R7 tren val HI-Medium (bo qua contrib), trong cua so rieng.
rem Nhap dup tep nay. Giu cua so mo, cam sac. Khong dung dau nhay kep long nhau trong lenh bash.
title FinGuard R7 bang xep hang - KHONG DONG CUA SO NAY
chcp 65001 >nul
"C:\Program Files\Git\bin\bash.exe" -c "cd '/d/Risk Management - HVTC Competition/finguard/Finguard' && echo === $(date +%%F_%%T) chay bang xep hang R7, SKIP_CONTRIB=1 >> logs/r7_chain.log && SKIP_CONTRIB=1 bash scripts/run_r7.sh 2>&1 | tee -a logs/r7_console.log; echo exit ${PIPESTATUS[0]} >> logs/r7_chain.log"
echo.
echo Xong phien nay. Nhan phim bat ky de dong cua so.
pause >nul
