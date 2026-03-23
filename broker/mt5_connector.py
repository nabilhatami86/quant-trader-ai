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
    "GOLD":   "XAUUSDm",
    "XAUUSD": "XAUUSDm",
    "XAUEUR": "XAUEURm",
    "EURUSD": "EURUSDm",
    "GBPUSD": "GBPUSDm",
    "USDJPY": "USDJPYm",
    "DXY":    "USDXm",
    "BTCUSD": "BTCUSDm",
}

DEFAULT_LOT  = 0.01
MAX_LOT      = 0.10
SLIPPAGE     = 10
MAGIC_NUMBER = 202601
BOT_COMMENT  = "TraderAI-Bot"
RISK_PERCENT = 1.0


class MT5Connector:
    def __init__(self):
        self.connected  = False
        self.account    = {}
        self.last_error = ""

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

    def calculate_auto_lot(self, min_lot: float = 0.01,
                           max_lot: float = 0.50) -> float:
        from config import REAL_AUTO_LOT_MIN, REAL_AUTO_LOT_MAX
        self._refresh_account()
        balance = self.account.get("balance", 0)
        if balance <= 0:
            return min_lot

        # setiap $100 = 0.01 lot
        raw  = balance / 10000.0
        step = 0.01
        lot  = round(round(raw / step) * step, 2)
        lot  = max(REAL_AUTO_LOT_MIN, min(REAL_AUTO_LOT_MAX, lot))

        GREEN = "\033[92m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        print(f"  {BOLD}[AUTO-LOT]{RESET} Saldo ${balance:,.2f} "
              f"→ Lot {GREEN}{BOLD}{lot}{RESET} "
              f"(range {REAL_AUTO_LOT_MIN}–{REAL_AUTO_LOT_MAX})")
        return lot

    def calculate_signal_lot(self, base_lot: float,
                              signal_score: float = 0,
                              ml_confidence: float = 0,
                              symbol_key: str = "",
                              min_lot: float = 0.01,
                              max_lot: float = 0.50) -> float:
        """
        Lot dinamis berdasarkan kekuatan sinyal:

        Base lot  → dari risk-based calculation (balance/SL)
        Multiplier dari score:
          score < 5        → 1.0x  (lemah — base saja)
          score 5–8        → 1.3x  (cukup bagus)
          score 8–12       → 1.7x  (bagus)
          score ≥ 12       → 2.0x  (sangat kuat)

        Bonus dari ML confidence:
          conf 75–84%      → +0.2x tambahan
          conf ≥ 85%       → +0.4x tambahan
          (hanya berlaku jika score ≥ 5)

        Safety check: lot tidak boleh melebihi 40% free margin.
        """
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"

        abs_score = abs(signal_score)

        # Score multiplier
        if abs_score >= 12:
            score_mult = 2.0
            tier = "SANGAT KUAT ≥12"
        elif abs_score >= 8:
            score_mult = 1.7
            tier = "KUAT 8-12"
        elif abs_score >= 5:
            score_mult = 1.3
            tier = "CUKUP 5-8"
        else:
            score_mult = 1.0
            tier = "LEMAH <5"

        # ML confidence bonus
        conf_bonus = 0.0
        if abs_score >= 5:
            if ml_confidence >= 85:
                conf_bonus = 0.4
            elif ml_confidence >= 75:
                conf_bonus = 0.2

        total_mult = score_mult + conf_bonus

        raw_lot = base_lot * total_mult

        # Snap ke step 0.01
        step    = 0.01
        raw_lot = round(round(raw_lot / step) * step, 2)

        # Margin safety check — jangan pakai lebih dari 40% free margin
        self._refresh_account()
        free_margin = self.account.get("margin_free", 0)
        if free_margin > 0 and symbol_key:
            si = self.get_symbol_info(symbol_key)
            if si:
                # Estimasi margin per lot (kasar: leverage-based)
                leverage  = self.account.get("leverage", 100)
                ask       = si.get("ask", 1)
                lot_size  = 100_000   # standar forex
                margin_per_lot = (ask * lot_size) / max(leverage, 1)
                max_affordable = (free_margin * 0.4) / max(margin_per_lot, 1)
                max_affordable = round(round(max_affordable / step) * step, 2)
                if max_affordable > 0:
                    raw_lot = min(raw_lot, max_affordable)

        final_lot = max(min_lot, min(max_lot, raw_lot))
        color = GREEN if final_lot > base_lot else YELLOW
        print(f"  {BOLD}[SIGNAL-LOT]{RESET} Score {signal_score:+.1f} ({tier}) "
              f"| ML {ml_confidence:.0f}% "
              f"| Mult ×{total_mult:.1f} "
              f"| Base {base_lot} → {color}{BOLD}{final_lot}{RESET} lot")
        return final_lot

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

    def calculate_sl_from_risk(self, symbol_key: str, direction: str,
                                price: float, lot: float,
                                risk_usd: float) -> float:
        """
        Hitung harga SL berdasarkan maksimal kerugian dalam dolar.
        sl_distance = risk_usd / (lot * pip_value_per_lot)
        """
        si = self.get_symbol_info(symbol_key)
        if not si or risk_usd <= 0 or lot <= 0:
            return 0.0

        tick_val  = si.get("tick_value", 1.0)
        tick_size = si.get("tick_size", 0.01)
        point     = si.get("point", 0.00001)
        digits    = si.get("digits", 5)

        pip_value = (tick_val / tick_size) * point * 10
        if pip_value <= 0:
            return 0.0

        sl_pips     = risk_usd / (lot * pip_value)
        sl_distance = sl_pips * point * 10

        if direction == "BUY":
            sl = round(price - sl_distance, digits)
        else:
            sl = round(price + sl_distance, digits)

        return sl

    def place_order(self, symbol_key: str, direction: str,
                    lot: float = None, sl: float = None, tp: float = None,
                    comment: str = BOT_COMMENT, silent: bool = False) -> dict:
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

        # Validasi SL — kalau SL di sisi yang salah dari entry, recalculate dari ATR
        digits = si.get("digits", 2)
        if sl:
            if direction == "BUY" and sl >= price:
                from config import ATR_MULTIPLIER_SL
                atr = abs(price - sl)   # pakai selisih sebagai estimasi ATR
                sl  = round(price - max(atr, si["point"] * 50), digits)
                print(f"[!] SL dikoreksi — SL lama di atas entry, dipindah ke {sl:.2f}")
            elif direction == "SELL" and sl <= price:
                from config import ATR_MULTIPLIER_SL
                atr = abs(sl - price)
                sl  = round(price + max(atr, si["point"] * 50), digits)
                print(f"[!] SL dikoreksi — SL lama di bawah entry, dipindah ke {sl:.2f}")

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

        print(f"[MT5] Kirim order → symbol:{symbol} dir:{direction} "
              f"lot:{lot} price:{price:.2f} sl:{request['sl']} tp:{request['tp']}")

        result = mt5.order_send(request)

        if result is None:
            return {"success": False, "error": str(mt5.last_error())}

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            if not silent:
                print(f"[OK] Order {direction} {lot} lot {symbol} @ {price:.5f}")
                print(f"     Ticket: #{result.order}  SL: {sl}  TP: {tp}")
            try:
                from data.trade_journal import log_entry
                log_entry(symbol_key, "", result.order, direction,
                          price, sl or 0, tp or 0, lot, comment)
            except Exception:
                pass
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

        pos    = positions[0]
        new_sl = sl if sl is not None else pos.sl
        new_tp = tp if tp is not None else pos.tp

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
            try:
                from data.trade_journal import log_exit
                pnl = pos.profit if hasattr(pos, "profit") else 0
                log_exit(pos.symbol, "", ticket, price, pnl)
            except Exception:
                pass
            return {"success": True, "ticket": ticket, "close_price": price}
        return {"success": False, "error": str(mt5.last_error())}

    def partial_close(self, ticket: int, close_pct: float = 50.0) -> dict:
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Posisi #{ticket} tidak ditemukan"}

        pos       = positions[0]
        symbol    = pos.symbol
        full_vol  = pos.volume
        close_vol = round(full_vol * close_pct / 100.0, 2)

        si        = mt5.symbol_info(symbol)
        min_lot   = si.volume_min if si else 0.01
        step      = si.volume_step if si else 0.01
        close_vol = max(min_lot, round(round(close_vol / step) * step, 2))
        close_vol = min(close_vol, full_vol)

        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price      = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(symbol).ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       close_vol,
            "type":         order_type,
            "position":     ticket,
            "price":        price,
            "deviation":    SLIPPAGE,
            "magic":        MAGIC_NUMBER,
            "comment":      f"PartialClose#{ticket}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            remaining = round(full_vol - close_vol, 2)
            print(f"[OK] Partial close #{ticket} — "
                  f"tutup {close_vol} lot ({close_pct:.0f}%), "
                  f"sisa {remaining} lot @ {price:.5f}")
            try:
                from data.trade_journal import log_exit
                log_exit(symbol, "", ticket, price, pos.profit, note="PARTIAL")
            except Exception:
                pass
            return {"success": True, "ticket": ticket,
                    "closed_vol": close_vol, "remaining_vol": remaining,
                    "price": price}
        err = str(mt5.last_error())
        print(f"[ERROR] Partial close gagal: {err}")
        return {"success": False, "error": err}

    def close_all(self, symbol_key: str = None) -> list:
        symbol    = SYMBOL_MAP.get(symbol_key, symbol_key) if symbol_key else None
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        results   = []
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
        return [self._fmt_position(p) for p in self._raw_positions(symbol_key)]

    def get_ohlcv(self, symbol_key: str, timeframe: str = "1h",
                  count: int = 1000) -> "pd.DataFrame":
        import pandas as pd

        if not self.connected:
            return pd.DataFrame()

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
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_history(self, days: int = 7) -> list:
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
            pc     = GREEN if p["profit"] >= 0 else RED
            dc     = GREEN if p["direction"] == "BUY" else RED
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


class SignalExecutor:

    def __init__(self, connector: MT5Connector, symbol_key: str,
                 trailing_pips: float = 0.0,
                 dca_minutes: float = 0.0,
                 bulk_orders: int = 1,
                 fixed_lot: float = None,
                 strict_mode: bool = False,
                 risk_per_trade: float = 0.0,
                 real_mode: bool = False,
                 ml_min_conf: float = 0.0):
        self.mt5            = connector
        self.symbol         = symbol_key
        self.trailing_pips  = trailing_pips
        self.dca_minutes    = dca_minutes
        self.bulk_orders    = bulk_orders
        self.fixed_lot      = fixed_lot
        self.strict_mode    = strict_mode
        self.risk_per_trade = risk_per_trade
        self.real_mode      = real_mode
        self.ml_min_conf    = ml_min_conf
        self.last_sig       = None
        self.last_tick      = 0
        self._tp1_done      = set()

    def should_trade(self, signal: dict, ml_pred: dict) -> bool:
        direction = signal.get("direction", "WAIT")

        if direction == "WAIT":
            return False

        if self.strict_mode:
            from config import MICRO_ML_CONF
            if not ml_pred or ml_pred.get("direction") == "WAIT":
                print(f"  [MICRO] Trade dibatalkan — ML tidak ada prediksi")
                return False
            if ml_pred.get("direction") != direction:
                print(f"  [MICRO] Trade dibatalkan — ML tidak sepakat: "
                      f"Signal={direction}, ML={ml_pred.get('direction')} "
                      f"({ml_pred.get('confidence', 0)}%)")
                return False
            conf = ml_pred.get("confidence", 0)
            if conf < MICRO_ML_CONF:
                print(f"  [MICRO] Trade dibatalkan — ML confidence {conf}% "
                      f"< {MICRO_ML_CONF}% (minimum untuk akun mikro)")
                return False
            if ml_pred.get("uncertain"):
                print(f"  [MICRO] Trade dibatalkan — ML uncertain")
                return False
            all_pos = self.mt5.get_all_positions(self.symbol)
            if all_pos:
                print(f"  [MICRO] Sudah ada {len(all_pos)} posisi aktif — "
                      f"tunggu tutup dulu")
                return False
            return True

        if self.real_mode:
            from config import REAL_ML_CONF, REAL_MAX_STACK
            min_conf = self.ml_min_conf or REAL_ML_CONF
            if not ml_pred or ml_pred.get("direction") == "WAIT":
                print(f"  [REAL] Trade dibatalkan — ML tidak ada prediksi")
                return False
            if ml_pred.get("direction") != direction:
                print(f"  [REAL] Trade dibatalkan — ML tidak sepakat: "
                      f"Signal={direction}, ML={ml_pred.get('direction')} "
                      f"({ml_pred.get('confidence', 0)}%)")
                return False
            conf = ml_pred.get("confidence", 0)
            if conf < min_conf:
                print(f"  [REAL] Trade dibatalkan — ML confidence {conf}% < {min_conf}%")
                return False
            all_pos = self.mt5.get_all_positions(self.symbol)
            if all_pos:
                same = [p for p in all_pos if p["direction"] == direction]
                if len(same) >= REAL_MAX_STACK:
                    print(f"  [REAL] Sudah {len(same)} posisi {direction} — "
                          f"batas tumpuk {REAL_MAX_STACK}x tercapai")
                    return False
                print(f"  [REAL] Posisi {direction}: {len(same)}/{REAL_MAX_STACK} — "
                      f"sinyal bagus, pasang lagi")
            return True

        if ml_pred and ml_pred.get("direction") not in (direction, "WAIT"):
            if not ml_pred.get("uncertain", False):
                print(f"[!] Trade dibatalkan - ML tidak sepakat: "
                      f"Signal={direction}, ML={ml_pred.get('direction')} "
                      f"(confidence: {ml_pred.get('confidence', 0)}%)")
                return False
            else:
                print(f"[~] ML berbeda tapi uncertain — lanjut berdasarkan signal")

        if self.dca_minutes > 0:
            elapsed = (time.time() - self.last_tick) / 60
            if self.last_tick > 0 and elapsed < self.dca_minutes:
                remaining = int(self.dca_minutes - elapsed)
                print(f"[~] DCA: tunggu {remaining} menit lagi sebelum order berikutnya")
                return False
            return True

        # Cek tumpuk posisi searah (general mode)
        from config import MAX_STACK
        all_pos = self.mt5.get_all_positions(self.symbol)
        if all_pos:
            same = [p for p in all_pos if p["direction"] == direction]
            if len(same) >= MAX_STACK:
                print(f"[~] Sudah {len(same)} posisi {direction} — "
                      f"batas tumpuk {MAX_STACK}x tercapai")
                return False
            print(f"[~] Posisi {direction}: {len(same)}/{MAX_STACK} — "
                  f"sinyal masih bagus, pasang lagi")

        return True

    def _resolve_sl(self, direction: str, price: float,
                    signal_sl: float, lot: float) -> float:
        """Jika risk_per_trade aktif, ganti SL dengan kalkulasi risk-based."""
        if self.risk_per_trade > 0 and lot and lot > 0:
            return self.mt5.calculate_sl_from_risk(
                self.symbol, direction, price, lot, self.risk_per_trade
            )
        return signal_sl

    def execute(self, signal: dict, ml_pred: dict = None,
                news_risk: str = "LOW") -> dict:
        if not self.mt5.connected:
            return {"success": False, "error": "MT5 tidak terhubung"}

        direction = signal.get("direction", "WAIT")
        sl        = signal.get("sl")
        tp        = signal.get("tp")

        if not self.should_trade(signal, ml_pred or {}):
            return {"success": False, "skipped": True}

        # Ambil harga sekarang dulu — dipakai untuk lot calc + SL resolve
        si    = self.mt5.get_symbol_info(self.symbol)
        price = si.get("ask" if direction == "BUY" else "bid", 0) if si else 0
        sl    = self._resolve_sl(direction, price, sl, self.fixed_lot or 0.01)

        # Signal strength info untuk dynamic lot
        sig_score  = float(signal.get("score", 0))
        ml_conf    = float(ml_pred.get("confidence", 0)) if ml_pred else 0.0

        lot = self.fixed_lot
        if self.real_mode:
            from config import (REAL_AUTO_LOT, REAL_AUTO_LOT_MIN,
                                REAL_AUTO_LOT_MAX, REAL_RISK_PCT)
            if REAL_AUTO_LOT:
                if sl and price and sl != price:
                    # Step 1: hitung base lot dari risk
                    sl_pips  = abs(price - sl) / (si.get("point", 0.01) * 10)
                    base_lot = self.mt5.calculate_lot(
                        self.symbol, sl_pips, REAL_RISK_PCT
                    )
                    base_lot = max(REAL_AUTO_LOT_MIN, min(REAL_AUTO_LOT_MAX, base_lot))

                    # Step 2: scale up berdasarkan kekuatan sinyal
                    lot = self.mt5.calculate_signal_lot(
                        base_lot,
                        signal_score=sig_score,
                        ml_confidence=ml_conf,
                        symbol_key=self.symbol,
                        min_lot=REAL_AUTO_LOT_MIN,
                        max_lot=REAL_AUTO_LOT_MAX,
                    )
                else:
                    # Tidak ada SL — pakai auto lot lalu scale
                    base_lot = self.mt5.calculate_auto_lot(REAL_AUTO_LOT_MIN, REAL_AUTO_LOT_MAX)
                    lot = self.mt5.calculate_signal_lot(
                        base_lot,
                        signal_score=sig_score,
                        ml_confidence=ml_conf,
                        symbol_key=self.symbol,
                        min_lot=REAL_AUTO_LOT_MIN,
                        max_lot=REAL_AUTO_LOT_MAX,
                    )

        n = self.bulk_orders
        if n > 1:

            success_tickets = []
            failed          = 0
            for _ in range(n):
                res = self.mt5.place_order(self.symbol, direction,
                                           lot=lot, sl=sl, tp=tp, silent=True)
                if res.get("success"):
                    success_tickets.append(res.get("ticket"))
                else:
                    failed += 1

            self.last_sig  = direction
            self.last_tick = time.time()

            GREEN = "\033[92m"
            RED   = "\033[91m"
            CYAN  = "\033[96m"
            BOLD  = "\033[1m"
            RESET = "\033[0m"
            dc    = GREEN if direction == "BUY" else RED

            lot_str    = f"{lot} lot/order" if lot else "auto lot"
            total_lots = round(lot * len(success_tickets), 2) if lot else "auto"
            risk_str   = (f"  Risk   : max ${self.risk_per_trade:.2f}/order "
                          f"= max ${self.risk_per_trade * len(success_tickets):.2f} total"
                          if self.risk_per_trade > 0 else "")

            print(f"\n  {'='*50}")
            print(f"  {BOLD}BULK ORDER — {dc}{direction}{RESET}{BOLD} × {len(success_tickets)}{RESET}")
            print(f"  {'='*50}")
            print(f"  Lot    : {lot_str}  (total: {total_lots} lot)")
            if risk_str:
                print(risk_str)
            print(f"  SL : {sl}   TP : {tp}")
            if success_tickets:
                id_range = (f"#{success_tickets[0]} — #{success_tickets[-1]}"
                            if len(success_tickets) > 1 else f"#{success_tickets[0]}")
                print(f"  Tickets : {CYAN}{id_range}{RESET}")
                print(f"  {GREEN}✓ {len(success_tickets)}/{n} order masuk{RESET}", end="")
                if failed:
                    print(f"  {RED}({failed} gagal){RESET}", end="")
                print()
            else:
                print(f"  {RED}✗ Semua order gagal{RESET}")
            print(f"  {'='*50}\n")

            return {
                "success":   bool(success_tickets),
                "bulk":      True,
                "count":     len(success_tickets),
                "tickets":   success_tickets,
                "direction": direction,
            }

        # Saat stacking: gunakan SL yang paling jauh dari harga saat ini
        # agar posisi lama tidak kena SL duluan
        existing = self.mt5.get_positions(self.symbol)
        same_dir = [p for p in existing if p["direction"] == direction]
        if same_dir and sl:
            existing_sls = [p["sl"] for p in same_dir if p.get("sl")]
            if existing_sls:
                if direction == "BUY":
                    # BUY: SL di bawah harga — pakai yg paling rendah (paling jauh)
                    sl = min(sl, min(existing_sls))
                else:
                    # SELL: SL di atas harga — pakai yg paling tinggi (paling jauh)
                    sl = max(sl, max(existing_sls))

                # Update SL posisi lama agar semua sejajar
                for pos in same_dir:
                    if pos.get("sl") != sl:
                        res = self.mt5.modify_position(pos["ticket"], sl=sl)
                        if res.get("success"):
                            print(f"  [STACK] #{pos['ticket']} SL disejajarkan → {sl:.5f}")

        result = self.mt5.place_order(
            symbol_key=self.symbol,
            direction=direction,
            lot=lot,
            sl=sl,
            tp=tp,
        )
        self.last_sig  = direction
        self.last_tick = time.time()
        return result

    def sync_manual_positions(self, signal: dict) -> None:
        manual    = self.mt5.get_manual_positions(self.symbol)
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

    def check_partial_tp(self) -> None:
        from config import MULTI_TP_ENABLED, TP1_RATIO, TP1_CLOSE_PCT
        if not MULTI_TP_ENABLED:
            return

        si = self.mt5.get_symbol_info(self.symbol)
        if not si:
            return

        GREEN = "\033[92m"
        CYAN  = "\033[96m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"

        bid = si["bid"]
        ask = si["ask"]

        for pos in self.mt5.get_all_positions(self.symbol):
            ticket    = pos["ticket"]
            if ticket in self._tp1_done:
                continue

            entry     = pos["open_price"]
            tp        = pos["tp"]
            direction = pos["direction"]

            if not tp or tp == 0:
                continue

            if direction == "BUY":
                tp1           = entry + (tp - entry) * TP1_RATIO
                current_price = bid
                hit_tp1       = current_price >= tp1
            else:
                tp1           = entry - (entry - tp) * TP1_RATIO
                current_price = ask
                hit_tp1       = current_price <= tp1

            if not hit_tp1:
                continue

            pnl = pos.get("profit", 0)
            print(f"\n  {BOLD}{CYAN}[TP1] #{ticket} {direction} — "
                  f"harga {current_price:.5f} mencapai TP1 {tp1:.5f}{RESET}")
            print(f"  {CYAN}→ Partial close {TP1_CLOSE_PCT:.0f}% posisi "
                  f"(P&L sementara: ${pnl:+.2f}){RESET}")

            res = self.mt5.partial_close(ticket, close_pct=TP1_CLOSE_PCT)
            if res.get("success"):
                self._tp1_done.add(ticket)
                from config import BREAKEVEN_BUFFER
                digits = si.get("digits", 5)
                if direction == "BUY":
                    be_sl = round(entry + abs(entry) * BREAKEVEN_BUFFER * 0.001, digits)
                else:
                    be_sl = round(entry - abs(entry) * BREAKEVEN_BUFFER * 0.001, digits)
                mod = self.mt5.modify_position(ticket, sl=be_sl)
                if mod.get("success"):
                    print(f"  {GREEN}✓ SL digeser ke breakeven {be_sl:.5f} "
                          f"— sisa {res['remaining_vol']} lot bebas risiko{RESET}\n")

    def check_breakeven(self) -> None:
        from config import BREAKEVEN_ENABLED, BREAKEVEN_TRIGGER, BREAKEVEN_BUFFER
        if not BREAKEVEN_ENABLED:
            return

        si = self.mt5.get_symbol_info(self.symbol)
        if not si:
            return

        for pos in self.mt5.get_positions(self.symbol):
            ticket     = pos["ticket"]
            direction  = pos["direction"]
            entry      = pos["open_price"]
            current_sl = pos["sl"]
            sl_dist    = abs(entry - current_sl) if current_sl else 0

            if sl_dist == 0:
                continue

            bid = si["bid"]
            ask = si["ask"]

            if direction == "BUY":
                current_price = bid
                profit_dist   = current_price - entry
                breakeven_sl  = round(entry + sl_dist * BREAKEVEN_BUFFER, si["digits"])
                if profit_dist >= sl_dist * BREAKEVEN_TRIGGER and current_sl < breakeven_sl:
                    res = self.mt5.modify_position(ticket, sl=breakeven_sl)
                    if res["success"]:
                        print(f"[BE] Breakeven aktif #{ticket} BUY — "
                              f"SL digeser ke {breakeven_sl:.5f}")

            elif direction == "SELL":
                current_price = ask
                profit_dist   = entry - current_price
                breakeven_sl  = round(entry - sl_dist * BREAKEVEN_BUFFER, si["digits"])
                if profit_dist >= sl_dist * BREAKEVEN_TRIGGER and (current_sl == 0 or current_sl > breakeven_sl):
                    res = self.mt5.modify_position(ticket, sl=breakeven_sl)
                    if res["success"]:
                        print(f"[BE] Breakeven aktif #{ticket} SELL — "
                              f"SL digeser ke {breakeven_sl:.5f}")

    def check_abnormal_movement(self, df=None) -> None:
        if df is None or df.empty or len(df) < 3:
            return

        si = self.mt5.get_symbol_info(self.symbol)
        if not si:
            return

        RED   = "\033[91m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"

        last   = df.iloc[-1]
        atr    = float(last.get("atr", 0))
        if atr <= 0:
            return

        body      = abs(float(last["Close"]) - float(last["Open"]))
        direction = "BULL" if float(last["Close"]) > float(last["Open"]) else "BEAR"

        # Candle dianggap abnormal jika body > 4x ATR (spike ekstrem saja)
        if body < atr * 4.0:
            return

        # Analisis candle anomali
        last_row  = df.iloc[-1]
        prev_row  = df.iloc[-2]
        high      = float(last_row["High"])
        low       = float(last_row["Low"])
        open_     = float(last_row["Open"])
        close_    = float(last_row["Close"])
        wick_up   = round(high - max(open_, close_), 2)
        wick_down = round(min(open_, close_) - low, 2)
        full_range = round(high - low, 2)
        body_pct  = round(body / full_range * 100, 1) if full_range > 0 else 0
        prev_body = abs(float(prev_row["Close"]) - float(prev_row["Open"]))
        body_vs_prev = round(body / prev_body, 1) if prev_body > 0 else 0
        rsi       = round(float(last_row.get("rsi", 50)), 1)

        anomaly_reasons = []
        if body >= atr * 4.0:
            anomaly_reasons.append(f"body {body:.2f} = {body/atr:.1f}x ATR (normal max ~2x)")
        if body_vs_prev >= 3:
            anomaly_reasons.append(f"body {body_vs_prev}x lebih besar dari candle sebelumnya")
        if body_pct >= 85:
            anomaly_reasons.append(f"hampir seluruh range adalah body ({body_pct}%) — momentum ekstrem")
        if wick_up < 0.1 * full_range and direction == "BEAR":
            anomaly_reasons.append("wick atas sangat kecil — tekanan jual dominan penuh")
        if wick_down < 0.1 * full_range and direction == "BULL":
            anomaly_reasons.append("wick bawah sangat kecil — tekanan beli dominan penuh")

        for pos in self.mt5.get_all_positions(self.symbol):
            ticket   = pos["ticket"]
            pos_dir  = pos["direction"]

            spike_against = (pos_dir == "BUY"  and direction == "BEAR") or \
                            (pos_dir == "SELL" and direction == "BULL")

            if not spike_against:
                continue

            pnl    = pos.get("profit", 0)
            entry  = pos.get("open_price", 0)
            sl     = pos.get("sl", 0)
            YELLOW = "\033[93m"

            print(f"\n  {'='*54}")
            print(f"  {BOLD}{RED}⚠  ANOMALI PERGERAKAN TERDETEKSI{RESET}")
            print(f"  {'='*54}")
            print(f"  Candle  : {direction}  O:{open_:.2f}  H:{high:.2f}  "
                  f"L:{low:.2f}  C:{close_:.2f}")
            print(f"  Body    : {body:.2f} pts  ({body/atr:.1f}x ATR)  "
                  f"WickU:{wick_up:.2f}  WickD:{wick_down:.2f}")
            print(f"  RSI     : {rsi}  {'↑ rising' if rsi > float(prev_row.get('rsi',50)) else '↓ falling'}")
            print(f"\n  {BOLD}Kenapa dianggap anomali:{RESET}")
            for i, reason in enumerate(anomaly_reasons, 1):
                print(f"    {YELLOW}{i}. {reason}{RESET}")
            print(f"\n  {BOLD}Dampak ke posisi #{ticket} {pos_dir}:{RESET}")
            print(f"    Entry  : {entry:.2f}")
            print(f"    SL     : {sl:.2f}  →  jarak {abs(entry-sl):.2f} pts dari entry")
            print(f"    P&L    : {RED if pnl < 0 else ''}{pnl:+.2f}{RESET}")
            print(f"\n  {RED}{BOLD}→ Posisi ditutup paksa sebelum SL kena{RESET}")
            print(f"  {'='*54}\n")

            self.mt5.close_position(ticket)

    def check_sl_risk(self) -> None:
        si = self.mt5.get_symbol_info(self.symbol)
        if not si:
            return

        YELLOW = "\033[93m"
        RED    = "\033[91m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"

        bid = si["bid"]
        ask = si["ask"]

        for pos in self.mt5.get_all_positions(self.symbol):
            ticket    = pos["ticket"]
            direction = pos["direction"]
            entry     = pos["open_price"]
            sl        = pos["sl"]

            if not sl or sl == 0:
                continue

            sl_dist = abs(entry - sl)
            if sl_dist == 0:
                continue

            if direction == "BUY":
                current_price = bid
                dist_to_sl    = current_price - sl
            else:
                current_price = ask
                dist_to_sl    = sl - current_price

            if dist_to_sl <= 0:
                continue

            ratio = dist_to_sl / sl_dist
            pnl   = pos.get("profit", 0)
            label = "" if pos.get("magic") == MAGIC_NUMBER else " [manual]"

            from config import SL_CRITICAL_PCT, SL_DANGER_PCT

            if ratio <= SL_CRITICAL_PCT:
                print(f"\n  {BOLD}{RED}[SL-KRITIS] #{ticket}{label} {direction} "
                      f"— harga {current_price:.5f} tinggal {ratio*100:.1f}% dari SL {sl:.5f}!{RESET}")
                print(f"  {RED}→ Menutup posisi otomatis (P&L: ${pnl:+.2f}){RESET}")
                self.mt5.close_position(ticket)

            elif ratio <= SL_DANGER_PCT:
                print(f"  {BOLD}{YELLOW}[SL-BAHAYA] #{ticket}{label} {direction} "
                      f"— {ratio*100:.0f}% sisa jarak ke SL "
                      f"| harga: {current_price:.5f}  SL: {sl:.5f} "
                      f"| P&L: ${pnl:+.2f}{RESET}")

    def manage_positions(self, signal: dict, df=None) -> None:
        direction = signal.get("direction", "WAIT")

        for pos in self.mt5.get_positions(self.symbol):
            if (pos["direction"] == "BUY"  and direction == "SELL") or \
               (pos["direction"] == "SELL" and direction == "BUY"):
                print(f"[~] Sinyal berbalik - menutup posisi #{pos['ticket']}")
                self.mt5.close_position(pos["ticket"])

        self.check_abnormal_movement(df)
        self.check_sl_risk()
        self.check_partial_tp()
        self.check_breakeven()
        self.sync_manual_positions(signal)

        if self.trailing_pips > 0:
            all_pos = self.mt5.get_all_positions(self.symbol)
            if all_pos:
                self.mt5.update_trailing_stop(self.symbol, self.trailing_pips,
                                              include_manual=True)
