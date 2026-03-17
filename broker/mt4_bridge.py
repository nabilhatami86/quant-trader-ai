"""
MetaTrader 4 Bridge (via File Signal)
======================================
Untuk MT4 yang tidak punya Python API resmi.
Cara kerja:
  1. Python bot nulis sinyal ke file signal.txt
  2. EA di MT4 baca file itu setiap detik dan eksekusi order

Cara setup:
  1. Copy file EA_TraderAI.mq4 ke folder: MT4/MQL4/Experts/
  2. Compile di MetaEditor
  3. Pasang EA ke chart EURUSD atau XAUUSD
  4. Jalankan bot dengan: python main.py --mt4
"""

import os
import json
import time
from datetime import datetime

# Path file signal - sesuaikan dengan folder MT4 Anda
MT4_SIGNAL_FILE = os.path.join(
    os.environ.get("APPDATA", ""),
    "MetaQuotes", "Terminal",
    "Common", "Files",
    "trader_ai_signal.txt"
)
# Atau path alternatif (uncomment jika diperlukan):
# MT4_SIGNAL_FILE = "C:/Users/muham/AppData/Roaming/MetaQuotes/Terminal/Common/Files/trader_ai_signal.txt"

MT4_LOG_FILE = MT4_SIGNAL_FILE.replace("signal.txt", "log.txt")


class MT4Bridge:
    def __init__(self, signal_path: str = None):
        self.signal_path = signal_path or MT4_SIGNAL_FILE
        self.log_path    = self.signal_path.replace("signal.txt", "log.txt")
        # Pastikan folder ada
        os.makedirs(os.path.dirname(self.signal_path), exist_ok=True)

    def write_signal(self, direction: str, symbol: str,
                     sl: float = 0.0, tp: float = 0.0,
                     lot: float = 0.01, comment: str = "TraderAI") -> bool:
        """Tulis sinyal ke file untuk dibaca EA MT4"""
        if direction not in ("BUY", "SELL", "CLOSE", "CLOSE_ALL"):
            return False

        signal = {
            "action":    direction,
            "symbol":    symbol,
            "lot":       lot,
            "sl":        sl,
            "tp":        tp,
            "comment":   comment,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "magic":     202601,
        }

        try:
            with open(self.signal_path, "w") as f:
                json.dump(signal, f)
            print(f"[OK] Sinyal {direction} {symbol} ditulis ke MT4 bridge")
            return True
        except Exception as e:
            print(f"[ERROR] Gagal tulis sinyal: {e}")
            return False

    def read_log(self) -> list:
        """Baca log eksekusi dari EA MT4"""
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
            return [l.strip() for l in lines[-20:]]  # 20 baris terakhir
        except Exception:
            return []

    def clear_signal(self):
        """Hapus sinyal setelah dieksekusi"""
        try:
            with open(self.signal_path, "w") as f:
                f.write('{"action":"NONE"}')
        except Exception:
            pass

    def generate_ea_code(self) -> str:
        """Generate kode Expert Advisor MT4 (.mq4)"""
        return '''//+------------------------------------------------------------------+
//|  TraderAI Bridge EA - MT4                                        |
//|  Baca sinyal dari Python bot dan eksekusi order                  |
//+------------------------------------------------------------------+
#property copyright "TraderAI"
#property version   "1.0"
#property strict

#include <json.mqh>   // Perlu library JSON (download dari MQL5 market)

// Parameter
extern int    CheckIntervalSec = 2;    // Interval cek sinyal (detik)
extern int    MagicNumber      = 202601;
extern double DefaultLot       = 0.01;
extern int    Slippage         = 10;

string SignalFile = "trader_ai_signal.txt";
string LogFile    = "trader_ai_log.txt";
datetime lastSignalTime = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("TraderAI Bridge EA started. Reading: ", TerminalInfoString(TERMINAL_COMMONDATA_PATH), "\\\\Files\\\\", SignalFile);
   EventSetTimer(CheckIntervalSec);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   CheckAndExecuteSignal();
}

void CheckAndExecuteSignal()
{
   // Baca file sinyal
   int handle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE) return;

   string content = "";
   while(!FileIsEnding(handle))
      content += FileReadString(handle);
   FileClose(handle);

   if(content == "" || content == "{\\"action\\":\\"NONE\\"}") return;

   // Parse JSON sederhana (manual parsing)
   string action  = ExtractField(content, "action");
   string symbol  = ExtractField(content, "symbol");
   string ts      = ExtractField(content, "timestamp");
   double lot     = StringToDouble(ExtractField(content, "lot"));
   double sl      = StringToDouble(ExtractField(content, "sl"));
   double tp      = StringToDouble(ExtractField(content, "tp"));
   string comment = ExtractField(content, "comment");

   if(lot <= 0) lot = DefaultLot;

   // Cek timestamp supaya tidak eksekusi sinyal lama
   if(ts == "") return;

   // Execute berdasarkan action
   int ticket = -1;
   if(action == "BUY")
   {
      double price = MarketInfo(symbol, MODE_ASK);
      ticket = OrderSend(symbol, OP_BUY, lot, price, Slippage, sl, tp, comment, MagicNumber, 0, clrGreen);
      if(ticket > 0)
         WriteLog("BUY " + symbol + " lot=" + DoubleToStr(lot) + " @ " + DoubleToStr(price) + " #" + IntegerToString(ticket));
      else
         WriteLog("ERROR BUY: " + IntegerToString(GetLastError()));
   }
   else if(action == "SELL")
   {
      double price = MarketInfo(symbol, MODE_BID);
      ticket = OrderSend(symbol, OP_SELL, lot, price, Slippage, sl, tp, comment, MagicNumber, 0, clrRed);
      if(ticket > 0)
         WriteLog("SELL " + symbol + " lot=" + DoubleToStr(lot) + " @ " + DoubleToStr(price) + " #" + IntegerToString(ticket));
      else
         WriteLog("ERROR SELL: " + IntegerToString(GetLastError()));
   }
   else if(action == "CLOSE_ALL")
   {
      CloseAllByMagic(symbol);
   }

   // Reset sinyal setelah dieksekusi
   int wh = FileOpen(SignalFile, FILE_WRITE|FILE_TXT|FILE_COMMON);
   if(wh != INVALID_HANDLE) {
      FileWriteString(wh, "{\\"action\\":\\"NONE\\"}");
      FileClose(wh);
   }
}

void CloseAllByMagic(string sym)
{
   for(int i = OrdersTotal()-1; i >= 0; i--) {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
         if(OrderMagicNumber() == MagicNumber && (sym == "" || OrderSymbol() == sym)) {
            if(OrderType() == OP_BUY)
               OrderClose(OrderTicket(), OrderLots(), MarketInfo(OrderSymbol(), MODE_BID), Slippage, clrWhite);
            else if(OrderType() == OP_SELL)
               OrderClose(OrderTicket(), OrderLots(), MarketInfo(OrderSymbol(), MODE_ASK), Slippage, clrWhite);
         }
      }
   }
   WriteLog("CLOSE_ALL executed for " + sym);
}

string ExtractField(string json, string key)
{
   string search = "\\"" + key + "\\":\\"";
   int start = StringFind(json, search);
   if(start < 0) {
      // Coba tanpa quotes (untuk angka)
      search = "\\"" + key + "\\":";
      start  = StringFind(json, search);
      if(start < 0) return "";
      start += StringLen(search);
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      if(end < 0) return "";
      return StringSubstr(json, start, end - start);
   }
   start += StringLen(search);
   int end = StringFind(json, "\\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

void WriteLog(string msg)
{
   int h = FileOpen(LogFile, FILE_WRITE|FILE_READ|FILE_TXT|FILE_COMMON);
   if(h != INVALID_HANDLE) {
      FileSeek(h, 0, SEEK_END);
      FileWriteString(h, TimeToStr(TimeCurrent()) + " | " + msg + "\\n");
      FileClose(h);
   }
}
//+------------------------------------------------------------------+
'''

    def save_ea_file(self, output_path: str = None) -> str:
        """Simpan file EA .mq4 ke disk"""
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "EA_TraderAI.mq4"
            )
        code = self.generate_ea_code()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[OK] EA file disimpan: {output_path}")
        print(f"     Copy file ini ke: MT4/MQL4/Experts/")
        print(f"     Compile di MetaEditor, lalu pasang ke chart")
        return output_path
