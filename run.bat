@echo off
chcp 65001 >nul
echo.
echo  ============================================
echo   TRADING ROBOT - EUR/USD and GOLD/USD
echo  ============================================
echo.
echo  -- AKURASI ML TINGGI (rekomendasi) ---------
echo  1. EURUSD 1d  [~94%% ML accuracy]
echo  2. GOLD   1d  [~60%% ML accuracy]
echo.
echo  -- ANALISIS INTRADAY -----------------------
echo  3. EURUSD 1h  [~51%% ML, rule-based lebih dominan]
echo  4. GOLD   1h  [~50%% ML, rule-based lebih dominan]
echo  5. EURUSD 15m
echo  6. GOLD   15m
echo.
echo  -- BACKTEST --------------------------------
echo  7. Backtest EURUSD (1h)
echo  8. Backtest GOLD   (1h)
echo.
echo  -- DENGAN LSTM (TensorFlow Deep Learning) --
echo  A. EURUSD 1d + LSTM
echo  B. GOLD   1d + LSTM
echo.
echo  -- META TRADER --------------------------------
echo  M1. Connect MT5 + Analisis EURUSD 1d
echo  M2. Connect MT5 + Live Mode EURUSD 1h
echo  M3. Generate EA file untuk MT4
echo  M4. MT4 Bridge + Live EURUSD 1h
echo  M5. Cek Status Akun MT5
echo.
echo  -- TOOLS -----------------------------------
echo  9. Live Mode EURUSD 1h (auto refresh 60s)
echo  0. Panduan Tuning Parameter
echo  C. Custom command
echo.
set /p CHOICE="Pilihan: "

if "%CHOICE%"=="1" python -X utf8 main.py --symbol EURUSD --tf 1d
if "%CHOICE%"=="2" python -X utf8 main.py --symbol GOLD   --tf 1d
if "%CHOICE%"=="3" python -X utf8 main.py --symbol EURUSD --tf 1h
if "%CHOICE%"=="4" python -X utf8 main.py --symbol GOLD   --tf 1h
if "%CHOICE%"=="5" python -X utf8 main.py --symbol EURUSD --tf 15m
if "%CHOICE%"=="6" python -X utf8 main.py --symbol GOLD   --tf 15m
if "%CHOICE%"=="7" python -X utf8 main.py --symbol EURUSD --tf 1h --backtest
if "%CHOICE%"=="8" python -X utf8 main.py --symbol GOLD   --tf 1h --backtest
if /i "%CHOICE%"=="A" python -X utf8 main.py --symbol EURUSD --tf 1d --lstm
if /i "%CHOICE%"=="B" python -X utf8 main.py --symbol GOLD   --tf 1d --lstm
if /i "%CHOICE%"=="M1" python -X utf8 main.py --symbol EURUSD --tf 1d --mt5
if /i "%CHOICE%"=="M2" python -X utf8 main.py --symbol EURUSD --tf 1h --mt5 --live
if /i "%CHOICE%"=="M3" python -X utf8 main.py --mt-setup
if /i "%CHOICE%"=="M4" python -X utf8 main.py --symbol EURUSD --tf 1h --mt4 --live
if /i "%CHOICE%"=="M5" python -X utf8 main.py --symbol EURUSD --tf 1h --mt5 --mt-status
if "%CHOICE%"=="9" python -X utf8 main.py --symbol EURUSD --tf 1h --live
if "%CHOICE%"=="0" python -X utf8 main.py --tune
if /i "%CHOICE%"=="C" (
    set /p ARGS="Masukkan argumen (contoh: --symbol EURUSD --tf 15m --backtest): "
    python -X utf8 main.py %ARGS%
)

pause
