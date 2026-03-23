//+------------------------------------------------------------------+
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
   Print("TraderAI Bridge EA started. Reading: ", TerminalInfoString(TERMINAL_COMMONDATA_PATH), "\\Files\\", SignalFile);
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

   if(content == "" || content == "{\"action\":\"NONE\"}") return;

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
      FileWriteString(wh, "{\"action\":\"NONE\"}");
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
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start < 0) {
      // Coba tanpa quotes (untuk angka)
      search = "\"" + key + "\":";
      start  = StringFind(json, search);
      if(start < 0) return "";
      start += StringLen(search);
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      if(end < 0) return "";
      return StringSubstr(json, start, end - start);
   }
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
}

void WriteLog(string msg)
{
   int h = FileOpen(LogFile, FILE_WRITE|FILE_READ|FILE_TXT|FILE_COMMON);
   if(h != INVALID_HANDLE) {
      FileSeek(h, 0, SEEK_END);
      FileWriteString(h, TimeToStr(TimeCurrent()) + " | " + msg + "\n");
      FileClose(h);
   }
}
//+------------------------------------------------------------------+
