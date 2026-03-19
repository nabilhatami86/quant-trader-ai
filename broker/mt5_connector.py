import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

MT5_CONFIG = {
    "path":     os.getenv("MT5_PATH", ""),
    "server":   os.getenv("MT5_SERVER", ""),
    "login":    int(os.getenv("MT5_LOGIN", "0")),
    "password": os.getenv("MT5_PASSWORD", ""),
    "timeout":  10000,
}

SYMBOL_MAP = {
    "EURUSD": "EURUSDm",
    "GOLD":   "XAUUSDm",
    "XAUUSD": "XAUUSDm",
}

DEFAULT_LOT  = 0.01
MAX_LOT      = 1
SLIPPAGE     = 10
MAGIC_NUMBER = 202601
BOT_COMMENT  = "TraderAI-Bot"
RISK_PERCENT = 1.0


class MT5Connector:
    def __init__(self):
        self.connected  = False
        self.account    = {}
        self.last_error = ""

    # ─── CONNECT ──────────────────────────────────────────────
    def connect(self, login: int = None, password: str = None,
                server: str = None, path: str = None) -> bool:
        if not MT5_AVAILABLE:
            print("[ERROR] MetaTrader5 library tidak terinstall.")
            print("        Jalankan: pip install MetaTrader5")
            return False

        login    = login    or MT5_CONFIG["login"]
        password = password or MT5_CONFIG["password"]
        server   = server   or MT5_CONFIG["server"]
        path     = path     or MT5_CONFIG["path"]

        kwargs = {"timeout": MT5_CONFIG["timeout"]}
        if path:
            kwargs["path"] = path

        if not mt5.initialize(**kwargs):
            self.last_error = f"MT5 initialize failed: {mt5.last_error()}"
            print(f"[ERROR] {self.last_error}")
            return False

        if login:
            ok = mt5.login(login=login, password=password, server=server)
            if not ok:
                self.last_error = f"MT5 login failed: {mt5.last_error()}"
                print(f"[ERROR] {self.last_error}")
                mt5.shutdown()
                return False

        self.connected = True
        self._refresh_account()
        print(f"[OK] MT5 Connected!")
        print(f"     Broker  : {self.account.get('company', 'Unknown')}")
        print(f"     Server  : {self.account.get('server', 'Unknown')}")
        print(f"     Account : {self.account.get('login', 0)}")
        print(f"     Balance : ${self.account.get('balance', 0):,.2f}")
        print(f"     Equity  : ${self.account.get('equity', 0):,.2f}")
        print(f"     Leverage: 1:{self.account.get('leverage', 0)}")
        return True

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            print("[OK] MT5 Disconnected")

    def _refresh_account(self):
        info = mt5.account_info()
        if info:
            self.account = info._asdict()

    # ─── SYMBOL INFO ──────────────────────────────────────────
    def get_symbol_info(self, symbol_key: str) -> dict:
        if not self.connected:
            return {}
        symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
        info   = mt5.symbol_info(symbol)
        if not info:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if not info:
            return {}
        return {
            "symbol":     symbol,
            "bid":        info.bid,
            "ask":        info.ask,
            "spread":     info.spread,
            "digits":     info.digits,
            "point":      info.point,
            "lot_min":    info.volume_min,
            "lot_max":    info.volume_max,
            "lot_step":   info.volume_step,
            "tick_value": info.trade_tick_value,
            "tick_size":  info.trade_tick_size,
        }

    def get_current_price(self, symbol_key: str) -> dict:
        si = self.get_symbol_info(symbol_key)
        return {
            "bid":    si.get("bid", 0),
            "ask":    si.get("ask", 0),
            "spread": si.get("spread", 0),
        }

    def calculate_lot(self, symbol_key: str, sl_pips: float,
                      risk_percent: float = RISK_PERCENT) -> float:
        self._refresh_account()
        balance  = self.account.get("balance", 10000)
        risk_usd = balance * (risk_percent / 100)
        si       = self.get_symbol_info(symbol_key)
        if not si or sl_pips <= 0:
            return DEFAULT_LOT

        tick_val  = si.get("tick_value", 1)
        tick_size = si.get("tick_size", 0.00001)
        pip_value = tick_val / tick_size * si.get("point", 0.00001) * 10

        if pip_value <= 0:
            return DEFAULT_LOT

        lot  = risk_usd / (sl_pips * pip_value)
        lot  = max(si.get("lot_min", 0.01), min(lot, MAX_LOT))
        step = si.get("lot_step", 0.01)
        lot  = round(round(lot / step) * step, 2)
        return lot

    def place_order(self, symbol_key: str, direction: str,
                    lot: float = None, sl: float = None, tp: float = None,
                    comment: str = BOT_COMMENT) -> dict:
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung ke MT5"}

        symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
        mt5.symbol_select(symbol, True)
        si = self.get_symbol_info(symbol_key)
        if not si:
            return {"success": False, "error": f"Symbol {symbol} tidak ditemukan"}

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price      = si["ask"]
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price      = si["bid"]

        if lot is None:
            sl_pips = abs(price - sl) / si["point"] / 10 if sl else 20
            lot     = self.calculate_lot(symbol_key, sl_pips)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           round(sl, si["digits"]) if sl else 0.0,
            "tp":           round(tp, si["digits"]) if tp else 0.0,
            "deviation":    SLIPPAGE,
            "magic":        MAGIC_NUMBER,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            return {"success": False, "error": str(mt5.last_error())}

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] Order {direction} {lot} lot {symbol} @ {price:.5f}")
            print(f"     Ticket: #{result.order}  SL: {sl}  TP: {tp}")
            return {
                "success":   True,
                "ticket":    result.order,
                "direction": direction,
                "symbol":    symbol,
                "lot":       lot,
                "price":     price,
                "sl":        sl,
                "tp":        tp,
                "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            err = f"Order gagal: retcode={result.retcode} ({result.comment})"
            print(f"[ERROR] {err}")
            return {"success": False, "error": err, "retcode": result.retcode}

    def modify_position(self, ticket: int,
                        sl: float = None, tp: float = None) -> dict:
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Posisi #{ticket} tidak ditemukan"}

        pos     = positions[0]
        new_sl  = sl if sl is not None else pos.sl
        new_tp  = tp if tp is not None else pos.tp

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       new_sl,
            "tp":       new_tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] Posisi #{ticket} diupdate — SL: {new_sl}  TP: {new_tp}")
            return {"success": True, "ticket": ticket, "sl": new_sl, "tp": new_tp}

        err = str(mt5.last_error())
        print(f"[ERROR] Modify gagal: {err}")
        return {"success": False, "error": err}

    def update_trailing_stop(self, symbol_key: str,
                             trail_pips: float = 20.0,
                             include_manual: bool = True) -> list:
        if not self.connected:
            return []

        si = self.get_symbol_info(symbol_key)
        if not si:
            return []

        point      = si.get("point", 0.00001)
        trail_dist = trail_pips * point * 10
        positions  = (self.get_all_positions(symbol_key) if include_manual
                      else self.get_positions(symbol_key))
        results    = []

        for pos in positions:
            ticket    = pos["ticket"]
            direction = pos["direction"]
            cur_sl    = pos["sl"]
            label     = "" if pos["magic"] == MAGIC_NUMBER else " [manual]"

            if direction == "BUY":
                bid    = si["bid"]
                new_sl = round(bid - trail_dist, si["digits"])
                if new_sl > cur_sl:
                    res = self.modify_position(ticket, sl=new_sl)
                    if res["success"]:
                        print(f"[~] Trailing BUY #{ticket}{label}: SL {cur_sl:.5f} → {new_sl:.5f}")
                    results.append(res)

            elif direction == "SELL":
                ask    = si["ask"]
                new_sl = round(ask + trail_dist, si["digits"])
                if cur_sl == 0 or new_sl < cur_sl:
                    res = self.modify_position(ticket, sl=new_sl)
                    if res["success"]:
                        print(f"[~] Trailing SELL #{ticket}{label}: SL {cur_sl:.5f} → {new_sl:.5f}")
                    results.append(res)

        return results

    def close_position(self, ticket: int) -> dict:
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Posisi #{ticket} tidak ditemukan"}

        pos    = positions[0]
        symbol = pos.symbol

        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price      = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(symbol).ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       pos.volume,
            "type":         order_type,
            "position":     ticket,
            "price":        price,
            "deviation":    SLIPPAGE,
            "magic":        MAGIC_NUMBER,
            "comment":      f"Close #{ticket}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] Posisi #{ticket} berhasil ditutup @ {price:.5f}")
            return {"success": True, "ticket": ticket, "close_price": price}
        return {"success": False, "error": str(mt5.last_error())}

    def close_all(self, symbol_key: str = None) -> list:
        symbol = SYMBOL_MAP.get(symbol_key, symbol_key) if symbol_key else None
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

        results = []
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    results.append(self.close_position(pos.ticket))
        return results

    def _raw_positions(self, symbol_key: str = None) -> list:
        if not self.connected:
            return []
        symbol = SYMBOL_MAP.get(symbol_key, symbol_key) if symbol_key else None
        raw    = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return list(raw) if raw else []

    def _fmt_position(self, p) -> dict:
        return {
            "ticket":     p.ticket,
            "symbol":     p.symbol,
            "direction":  "BUY" if p.type == 0 else "SELL",
            "lot":        p.volume,
            "open_price": p.price_open,
            "sl":         p.sl,
            "tp":         p.tp,
            "profit":     p.profit,
            "magic":      p.magic,
            "time":       datetime.fromtimestamp(p.time).strftime("%Y-%m-%d %H:%M"),
            "comment":    p.comment,
        }

    def get_positions(self, symbol_key: str = None) -> list:
        return [self._fmt_position(p) for p in self._raw_positions(symbol_key)
                if p.magic == MAGIC_NUMBER]

    def get_manual_positions(self, symbol_key: str = None) -> list:
        return [self._fmt_position(p) for p in self._raw_positions(symbol_key)
                if p.magic != MAGIC_NUMBER]

    def get_all_positions(self, symbol_key: str = None) -> list:
        """Semua posisi tanpa filter magic."""
        return [self._fmt_position(p) for p in self._raw_positions(symbol_key)]

    # ─── AMBIL DATA OHLCV LANGSUNG DARI MT5 ───────────────────
    def get_ohlcv(self, symbol_key: str, timeframe: str = "1h",
                  count: int = 1000) -> "pd.DataFrame":
        import pandas as pd
        import datetime as dt

        if not self.connected:
            return pd.DataFrame()

        # Mapping timeframe string → konstanta MT5
        TF_MAP = {
            "1m":  mt5.TIMEFRAME_M1,
            "5m":  mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h":  mt5.TIMEFRAME_H1,
            "4h":  mt5.TIMEFRAME_H4,
            "1d":  mt5.TIMEFRAME_D1,
            "1w":  mt5.TIMEFRAME_W1,
        }
        tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_H1)

        symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
        mt5.symbol_select(symbol, True)

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            print(f"[ERROR] MT5 get_ohlcv gagal: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.rename(columns={
            "open":        "Open",
            "high":        "High",
            "low":         "Low",
            "close":       "Close",
            "tick_volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df

    def get_history(self, days: int = 7) -> list:
        """Ambil history trade bot."""
        import datetime as dt
        from_date = dt.datetime.now() - dt.timedelta(days=days)
        deals     = mt5.history_deals_get(from_date, dt.datetime.now())
        if not deals:
            return []
        result = []
        for d in deals:
            if d.magic != MAGIC_NUMBER:
                continue
            result.append({
                "ticket": d.ticket,
                "symbol": d.symbol,
                "type":   "BUY" if d.type == 0 else "SELL",
                "lot":    d.volume,
                "price":  d.price,
                "profit": d.profit,
                "time":   datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M"),
            })
        return result

    # ─── PRINT STATUS ─────────────────────────────────────────
    def print_status(self) -> None:
        if not self.connected:
            print("[!] Tidak terhubung ke MT5")
            return

        self._refresh_account()
        bot_pos    = self.get_positions()
        manual_pos = self.get_manual_positions()

        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        sep    = "=" * 55

        def _print_pos(p, label=""):
            pc = GREEN if p["profit"] >= 0 else RED
            dc = GREEN if p["direction"] == "BUY" else RED
            sl_str = f"SL:{p['sl']:.2f}" if p["sl"] else "SL:--"
            tp_str = f"TP:{p['tp']:.2f}" if p["tp"] else "TP:--"
            print(f"    #{p['ticket']} {dc}{p['direction']}{RESET} {p['symbol']} "
                  f"{p['lot']}lot @ {p['open_price']:.2f} | "
                  f"{sl_str} {tp_str} | "
                  f"P&L: {pc}${p['profit']:+.2f}{RESET}{label}")

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  MT5 ACCOUNT STATUS")
        print(sep)
        print(f"  Broker  : {self.account.get('company', 'Unknown')}")
        print(f"  Account : {self.account.get('login', 0)}  ({self.account.get('server', '')})")
        print(f"  Balance : {BOLD}${self.account.get('balance', 0):,.2f}{RESET}")
        print(f"  Equity  : ${self.account.get('equity', 0):,.2f}")
        margin_free = self.account.get("margin_free", 0)
        print(f"  Margin  : ${self.account.get('margin', 0):,.2f}  Free: ${margin_free:,.2f}")
        profit = self.account.get("profit", 0)
        pc = GREEN if profit >= 0 else RED
        print(f"  Float P&L: {pc}{BOLD}${profit:+,.2f}{RESET}")
        print(sep)

        if bot_pos:
            print(f"  {BOLD}Bot Positions ({len(bot_pos)}):{RESET}")
            for p in bot_pos:
                _print_pos(p)
        else:
            print(f"  {YELLOW}Tidak ada posisi bot aktif{RESET}")

        if manual_pos:
            print(f"  {BOLD}{CYAN}Manual Positions ({len(manual_pos)}):{RESET}")
            for p in manual_pos:
                _print_pos(p, f"  {CYAN}[manual]{RESET}")
        print(sep)


# ─── SIGNAL EXECUTOR ──────────────────────────────────────
class SignalExecutor:

    def __init__(self, connector: MT5Connector, symbol_key: str,
                 trailing_pips: float = 0.0,
                 dca_minutes: float = 0.0):
        self.mt5           = connector
        self.symbol        = symbol_key
        self.trailing_pips = trailing_pips
        self.dca_minutes   = dca_minutes   # 0 = DCA off, >0 = jeda menit antar order
        self.last_sig      = None
        self.last_tick     = 0             # timestamp order terakhir

    def should_trade(self, signal: dict, ml_pred: dict) -> bool:
        direction = signal.get("direction", "WAIT")

        if direction == "WAIT":
            return False

        if ml_pred and ml_pred.get("direction") not in (direction, "WAIT"):
            if not ml_pred.get("uncertain", False):
                print(f"[!] Trade dibatalkan - ML tidak sepakat: "
                      f"Signal={direction}, ML={ml_pred.get('direction')} "
                      f"(confidence: {ml_pred.get('confidence', 0)}%)")
                return False
            else:
                print(f"[~] ML berbeda tapi uncertain — lanjut berdasarkan signal")

        # Jika DCA aktif: cek jeda waktu sejak order terakhir
        if self.dca_minutes > 0:
            elapsed = (time.time() - self.last_tick) / 60
            if self.last_tick > 0 and elapsed < self.dca_minutes:
                remaining = int(self.dca_minutes - elapsed)
                print(f"[~] DCA: tunggu {remaining} menit lagi sebelum order berikutnya")
                return False
            return True

        # Mode normal: blok jika sudah ada posisi searah
        positions = self.mt5.get_positions(self.symbol)
        if positions:
            existing = positions[0]["direction"]
            if existing == direction:
                print(f"[~] Sudah ada posisi {direction} aktif, skip")
                return False

        return True

    def execute(self, signal: dict, ml_pred: dict = None,
                news_risk: str = "LOW") -> dict:
        if not self.mt5.connected:
            return {"success": False, "error": "MT5 tidak terhubung"}

        direction = signal.get("direction", "WAIT")
        sl        = signal.get("sl")
        tp        = signal.get("tp")

        if not self.should_trade(signal, ml_pred or {}):
            return {"success": False, "skipped": True}

        result = self.mt5.place_order(
            symbol_key=self.symbol,
            direction=direction,
            sl=sl,
            tp=tp,
        )
        self.last_sig  = direction
        self.last_tick = time.time()
        return result

    def sync_manual_positions(self, signal: dict) -> None:
        manual = self.mt5.get_manual_positions(self.symbol)
        if not manual:
            return

        direction = signal.get("direction", "WAIT")
        sl        = signal.get("sl")
        tp        = signal.get("tp")

        for pos in manual:
            needs_sl = pos["sl"] == 0 and sl
            needs_tp = pos["tp"] == 0 and tp

            if not (needs_sl or needs_tp):
                continue

            # Hanya sync jika arah posisi cocok dengan sinyal (atau sinyal WAIT)
            if direction not in (pos["direction"], "WAIT"):
                continue

            res = self.mt5.modify_position(
                pos["ticket"],
                sl=sl if needs_sl else None,
                tp=tp if needs_tp else None,
            )
            if res["success"]:
                print(f"[MT5] Sync manual #{pos['ticket']} {pos['direction']} — "
                      f"SL: {sl}  TP: {tp}")

    def manage_positions(self, signal: dict) -> None:
        direction = signal.get("direction", "WAIT")

        # Close posisi bot yang berlawanan sinyal
        for pos in self.mt5.get_positions(self.symbol):
            if (pos["direction"] == "BUY"  and direction == "SELL") or \
               (pos["direction"] == "SELL" and direction == "BUY"):
                print(f"[~] Sinyal berbalik - menutup posisi #{pos['ticket']}")
                self.mt5.close_position(pos["ticket"])

        # Sync SL/TP ke posisi manual
        self.sync_manual_positions(signal)

        # Trailing stop untuk semua posisi (bot + manual)
        if self.trailing_pips > 0:
            all_pos = self.mt5.get_all_positions(self.symbol)
            if all_pos:
                self.mt5.update_trailing_stop(self.symbol, self.trailing_pips,
                                              include_manual=True)
