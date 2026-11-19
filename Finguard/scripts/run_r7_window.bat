@echo off
rem Chay R7 them 1 epoch trong cua so rieng: dong app Claude KHONG lam chet tien trinh train.
rem Nhap dup tep nay (hoac: explorer.exe scripts\run_r7_window.bat). Giu cua so mo, cam sac.
rem Lenh bash ben duoi KHONG duoc chua dau nhay kep long nhau (cmd se hieu sai >> va |).
title FinGuard R7 - DANG TRAIN, KHONG DONG CUA SO NAY
chcp 65001 >nul
"C:\Program Files\Git\bin\bash.exe" -c "cd '/d/Risk Management - HVTC Competition/finguard/Finguard' && echo === $(date +%%F_%%T) chay R7 trong cua so rieng, STOP_AFTER=1 >> logs/r7_chain.log && STOP_AFTER=1 bash scripts/run_r7.sh 2>&1 | tee -a logs/r7_console.log; echo exit ${PIPESTATUS[0]} >> logs/r7_chain.log"
echo.
echo Xong phien nay. Nhan phim bat ky de dong cua so.
pause >nul
