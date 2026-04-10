@echo off
chcp 65001 >nul
cls
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║        TRADING ROBOT  -  EUR/USD  ^&  GOLD/USD        ║
echo  ║        Powered by Yahoo Finance + ML + MT5           ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo  ─── API SERVER (rekomendasi) ────────────────────────────
echo  1. Jalankan API + Bot XAUUSD  [REAL MODE, MT5 auto-order]
echo  2. Jalankan API + Bot XAUUSD  [DEBUG mode, reload otomatis]
echo.
echo  ─── CLI LIVE MODE ───────────────────────────────────────
echo  3. XAUUSD 1h  --live --mt5 --real
echo  4. XAUUSD 5m  --live --mt5 --real
echo  5. XAUUSD 1h  --live --mt5 --micro
echo  6. EURUSD 1h  --live --mt5 --real
echo.
echo  ─── ANALISIS SEKALI ─────────────────────────────────────
echo  7. XAUUSD 1h  (analisis + MT5 status)
echo  8. XAUUSD 5m
echo  9. EURUSD 1h
echo.
echo  ─── BACKTEST ────────────────────────────────────────────
echo  B1. Backtest XAUUSD 1h
echo  B2. Backtest EURUSD 1h
echo.
echo  ─── TOOLS ───────────────────────────────────────────────
echo  D1. Backfill data XAUUSD ke PostgreSQL
echo  D2. Cek status akun MT5
echo  D3. Panduan tuning parameter
echo   C. Custom command
echo.
set /p CHOICE="Pilihan: "

if "%CHOICE%"=="1"  uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 1
if "%CHOICE%"=="2"  uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

if "%CHOICE%"=="3"  python -X utf8 main.py --symbol XAUUSD --tf 1h  --live --mt5 --real
if "%CHOICE%"=="4"  python -X utf8 main.py --symbol XAUUSD --tf 5m  --live --mt5 --real
if "%CHOICE%"=="5"  python -X utf8 main.py --symbol XAUUSD --tf 1h  --live --mt5 --micro
if "%CHOICE%"=="6"  python -X utf8 main.py --symbol EURUSD --tf 1h  --live --mt5 --real

if "%CHOICE%"=="7"  python -X utf8 main.py --symbol XAUUSD --tf 1h  --mt5 --mt-status
if "%CHOICE%"=="8"  python -X utf8 main.py --symbol XAUUSD --tf 5m
if "%CHOICE%"=="9"  python -X utf8 main.py --symbol EURUSD --tf 1h

if /i "%CHOICE%"=="B1" python -X utf8 main.py --symbol XAUUSD --tf 1h --backtest
if /i "%CHOICE%"=="B2" python -X utf8 main.py --symbol EURUSD --tf 1h --backtest

if /i "%CHOICE%"=="D1" python -X utf8 scripts/backfill.py --symbol XAUUSD
if /i "%CHOICE%"=="D2" python -X utf8 main.py --symbol XAUUSD --tf 1h --mt5 --mt-status
if /i "%CHOICE%"=="D3" python -X utf8 main.py --tune

if /i "%CHOICE%"=="C" (
    set /p ARGS="Masukkan argumen (contoh: --symbol XAUUSD --tf 1h --backtest): "
    python -X utf8 main.py %ARGS%
)

echo.
pause
