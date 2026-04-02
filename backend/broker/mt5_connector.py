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

        # Timeout via threading (Windows-safe, tidak pakai SIGALRM)
        import threading as _thr
        _init_result  = [None]
        _timeout_secs = MT5_CONFIG["timeout"] / 1000 + 10  # ms → s + buffer

        def _do_init():
            try:
                _init_result[0] = mt5.initialize(**kwargs)
            except Exception as _e:
                _init_result[0] = False

        _t = _thr.Thread(target=_do_init, daemon=True)
        _t.start()
        _t.join(timeout=_timeout_secs)

        if _t.is_alive():
            self.last_error = f"MT5 initialize() timeout setelah {_timeout_secs:.0f}s"
            print(f"[ERROR] {self.last_error}")
            return False

        if not _init_result[0]:
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
                # _journal_tf diset dari SignalExecutor sebelum order
                _tf = getattr(self, "_journal_tf", "")
                log_entry(symbol_key, _tf, result.order, direction,
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
                  count: int = 1000, include_forming: bool = True) -> "pd.DataFrame":
        """
        Ambil data OHLCV dari MT5.

        include_forming=True  → sertakan candle yang sedang terbentuk (realtime live)
                                 df.iloc[-1] = candle LIVE (harga bergerak)
        include_forming=False → hanya candle yang sudah CLOSED
                                 df.iloc[-1] = candle terakhir yang sudah tutup
        """
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

        fetch_count = count if include_forming else count + 1
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, fetch_count)
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

        if not include_forming:
            # Buang candle forming (baris terakhir = posisi 0 di MT5)
            df = df.iloc[:-1]

        return df

    def is_market_open(self, symbol_key: str) -> bool:
        """
        Cek apakah market symbol sedang buka (bisa ditrade).
        Return False jika market tutup (weekend, holiday, atau session tutup).
        """
        if not self.connected:
            return False
        try:
            symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
            info   = mt5.symbol_info(symbol)
            if not info:
                return False
            # trade_mode 0 = SYMBOL_TRADE_MODE_DISABLED (tutup)
            return info.trade_mode != 0
        except Exception:
            return True  # Kalau error, asumsikan buka agar tidak blok bot

    def get_realtime_data(self, symbol_key: str, tick_seconds: int = 60) -> dict:
        """
        Kumpulkan data realtime dari MT5 untuk memperkuat analisis:
          - tick_momentum : arah pergerakan harga dalam N detik terakhir
          - orderbook     : tekanan beli vs jual dari order yang antri (DOM)
          - spread        : lebar spread saat ini vs normal
          - current_tick  : harga bid/ask terkini

        Return: {
          "tick_momentum": {"direction", "bull_ratio", "up_ticks", "down_ticks",
                            "tick_count", "price_change", "score"},
          "orderbook":     {"imbalance", "bias", "bid_vol", "ask_vol",
                            "top_bids", "top_asks", "score"},
          "spread":        {"current", "normal", "ratio", "wide"},
          "current_tick":  {"bid", "ask", "last", "spread"},
          "realtime_score": float,   # -3 s/d +3, positif=bullish
          "realtime_bias":  str,     # BUY / SELL / NEUTRAL
        }
        """
        import pandas as pd
        from datetime import datetime, timedelta, timezone

        if not self.connected:
            return {}

        symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
        mt5.symbol_select(symbol, True)

        result = {}

        # ── 1. Current Tick (bid/ask/last) ───────────────────────────
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            result["current_tick"] = {
                "bid":    tick.bid,
                "ask":    tick.ask,
                "last":   tick.last,
                "spread": round(tick.ask - tick.bid, 5),
                "time":   datetime.fromtimestamp(tick.time, tz=timezone.utc),
            }
        else:
            result["current_tick"] = {}

        # ── 2. Tick Momentum (60 detik terakhir) ─────────────────────
        try:
            now      = datetime.now(timezone.utc)
            from_dt  = now - timedelta(seconds=tick_seconds)
            ticks    = mt5.copy_ticks_range(
                symbol, from_dt, now, mt5.COPY_TICKS_ALL
            )
            if ticks is not None and len(ticks) >= 5:
                df_t   = pd.DataFrame(ticks)
                prices = df_t["bid"].values   # pakai bid sebagai harga acuan
                up   = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
                down = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
                total = up + down
                bull_ratio = (up / total) if total > 0 else 0.5
                direction  = ("BUY"  if bull_ratio > 0.58 else
                              "SELL" if bull_ratio < 0.42 else "NEUTRAL")
                # Score -1.5 s/d +1.5
                tick_score = round((bull_ratio - 0.5) * 3.0, 2)
                result["tick_momentum"] = {
                    "direction":    direction,
                    "bull_ratio":   round(bull_ratio, 3),
                    "up_ticks":     up,
                    "down_ticks":   down,
                    "tick_count":   len(prices),
                    "price_change": round(float(prices[-1] - prices[0]), 5),
                    "score":        tick_score,
                }
            else:
                result["tick_momentum"] = {"direction": "NEUTRAL", "score": 0}
        except Exception:
            result["tick_momentum"] = {"direction": "NEUTRAL", "score": 0}

        # ── 3. Order Book / DOM ───────────────────────────────────────
        try:
            added = mt5.market_book_add(symbol)
            book  = mt5.market_book_get(symbol) if added else None
            if added:
                mt5.market_book_release(symbol)
            if book:
                bid_entries = [(b.price, b.volume) for b in book
                               if b.type == mt5.BOOK_TYPE_BUY]
                ask_entries = [(b.price, b.volume) for b in book
                               if b.type == mt5.BOOK_TYPE_SELL]
                bid_vol = sum(v for _, v in bid_entries)
                ask_vol = sum(v for _, v in ask_entries)
                total_v = bid_vol + ask_vol
                imbalance = ((bid_vol - ask_vol) / total_v
                             if total_v > 0 else 0.0)
                ob_bias  = ("BUY"  if imbalance >  0.15 else
                            "SELL" if imbalance < -0.15 else "NEUTRAL")
                ob_score = round(imbalance * 1.5, 2)   # -1.5 s/d +1.5
                result["orderbook"] = {
                    "imbalance": round(imbalance, 3),
                    "bias":      ob_bias,
                    "bid_vol":   round(bid_vol, 2),
                    "ask_vol":   round(ask_vol, 2),
                    "top_bids":  bid_entries[:3],
                    "top_asks":  ask_entries[:3],
                    "score":     ob_score,
                }
            else:
                result["orderbook"] = {"bias": "NEUTRAL", "score": 0,
                                       "imbalance": 0}
        except Exception:
            result["orderbook"] = {"bias": "NEUTRAL", "score": 0,
                                   "imbalance": 0}

        # ── 4. Spread Analysis ────────────────────────────────────────
        try:
            si = mt5.symbol_info(symbol)
            if si:
                spread_now    = tick.ask - tick.bid if tick else si.spread * si.point
                spread_normal = si.spread_float * si.point if si.spread_float > 0 \
                                else spread_now
                spread_ratio  = spread_now / spread_normal if spread_normal > 0 else 1.0
                wide_spread   = spread_ratio > 2.0
                result["spread"] = {
                    "current": round(spread_now, 5),
                    "normal":  round(spread_normal, 5),
                    "ratio":   round(spread_ratio, 2),
                    "wide":    wide_spread,
                }
            else:
                result["spread"] = {"wide": False, "ratio": 1.0}
        except Exception:
            result["spread"] = {"wide": False, "ratio": 1.0}

        # ── 5. Gabungkan score ────────────────────────────────────────
        tick_sc = result.get("tick_momentum", {}).get("score", 0)
        ob_sc   = result.get("orderbook",     {}).get("score", 0)
        wide    = result.get("spread",        {}).get("wide",  False)

        realtime_score = round(tick_sc + ob_sc, 2)
        if wide:
            realtime_score *= 0.5   # spread melebar → kurangi keyakinan

        result["realtime_score"] = realtime_score
        result["realtime_bias"]  = ("BUY"  if realtime_score >  0.5 else
                                    "SELL" if realtime_score < -0.5 else "NEUTRAL")
        return result

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
        self.last_sig            = None
        self.last_tick           = 0
        self._tp1_done           = set()
        self._sl_cooldown        = 0
        self._daily_loss         = 0.0
        self._daily_profit       = 0.0
        self._daily_date         = None
        self._daily_limit         = 0.0   # dihitung ulang tiap hari dari balance
        self._daily_start_balance = 0.0   # balance awal hari ini
        self._weekly_loss        = 0.0    # akumulasi loss minggu ini
        self._weekly_date        = None   # tanggal awal minggu saat ini
        self._known_tickets: set = set()   # tiket yang sudah kita catat OPEN
        self._timeframe          = ""      # diisi dari luar agar log_entry ada TF
        # Anti-overtrading state
        self._last_trade_time    = 0.0     # epoch seconds saat terakhir trade masuk
        self._last_sl_time       = 0.0     # epoch seconds saat terakhir kena SL
        self._trades_this_hour   = []      # list epoch seconds trade dalam jam ini
        self._trades_today       = 0       # counter trade hari ini
        self._trades_today_date  = None    # tanggal counter di atas
        # Loss Control System (Upgrade 8)
        self._loss_streak        = 0       # counter loss beruntun
        self._paused_until       = 0.0     # epoch saat pause berakhir
        # Confirmation Delay (Upgrade 3)
        self._pending_signal     = {}      # {"direction": "BUY", "time": epoch}
        # Delayed SL — pasang SL setelah beberapa menit, bukan langsung
        self._pending_sl: dict   = {}      # {ticket: {"sl": val, "tp": val, "direction": str, "placed_at": float}}

    def should_trade(self, signal: dict, ml_pred: dict) -> bool:
        direction = signal.get("direction", "WAIT")

        if direction == "WAIT":
            # Clear pending signal when market goes WAIT
            self._pending_signal = {}
            return False

        # ══════════════════════════════════════════════════════════════════
        # LOSS CONTROL SYSTEM — dinonaktifkan

        # ══════════════════════════════════════════════════════════════════
        # ADAPTIVE ENTRY TIMING (Upgrade)
        # STRONG  → entry langsung (no delay)
        # MEDIUM  → tunggu 1 candle konfirmasi (anti false signal)
        # WEAK    → skip entry
        # ══════════════════════════════════════════════════════════════════
        try:
            from config import CONFIRM_DELAY_ENABLED
            if CONFIRM_DELAY_ENABLED:
                _strength = signal.get("signal_strength", "MEDIUM")
                _spts     = signal.get("strength_pts", 0)

                if _strength == "STRONG":
                    # Setup sangat kuat → entry langsung, no delay
                    self._pending_signal = {}
                    print(f"  [ENTRY] {direction} STRONG ({_spts}/8 pts) → entry langsung ⚡")
                    # fallthrough ke anti-overtrading guards

                elif _strength == "MEDIUM":
                    # Setup cukup → tunggu 1 candle konfirmasi
                    _tf_sec = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(
                        getattr(self, "_timeframe", "5m"), 300)
                    _pend = self._pending_signal
                    if _pend.get("direction") == direction:
                        _elapsed = _now_check - _pend.get("time", 0)
                        if _elapsed >= _tf_sec:
                            self._pending_signal = {}   # konfirmasi OK, lanjut
                            print(f"  [ENTRY] {direction} MEDIUM — konfirmasi 1 candle OK ✓")
                        else:
                            _rem = int(_tf_sec - _elapsed)
                            print(f"  [ENTRY] {direction} MEDIUM ({_spts}/8 pts) — tunggu {_rem}s lagi")
                            return False
                    else:
                        self._pending_signal = {"direction": direction, "time": _now_check}
                        print(f"  [ENTRY] {direction} MEDIUM ({_spts}/8 pts) — pending, tunggu 1 candle")
                        return False

                else:
                    # WEAK setup → skip
                    # self._pending_signal = {}
                    # print(f"  [ENTRY] {direction} WEAK ({_spts}/8 pts) — setup terlalu lemah, skip")
                    # return False
                    pass  # WEAK tidak diblok, lanjut ke entry

        except Exception:
            pass

        # ══════════════════════════════════════════════════════════════════
        # ANTI-OVERTRADING GUARDS (berlaku semua mode)
        # ══════════════════════════════════════════════════════════════════
        import datetime as _dt
        import time as _time
        from config import (TRADE_COOLDOWN_MIN, SL_COOLDOWN_MIN,
                            MAX_TRADES_PER_HOUR, MAX_DAILY_TRADES)

        now_ts   = _time.time()
        now_date = _dt.date.today()

        # ── Reset daily counter jika hari baru ────────────────────────────
        if self._trades_today_date != now_date:
            self._trades_today_date = now_date
            self._trades_today      = 0

        # ── Max trades per hari (0 = disabled) ───────────────────────────
        if MAX_DAILY_TRADES > 0 and self._trades_today >= MAX_DAILY_TRADES:
            print(f"  [GUARD] Trade hari ini: {self._trades_today}/{MAX_DAILY_TRADES} "
                  f"— batas harian tercapai, berhenti sampai besok")
            return False

        _cooldown_min    = 0
        _max_trades_hour = 9999

        # ── Daily Profit limit (stop saat target tercapai, loss tetap jalan) ──
        try:
            from config import REAL_DAILY_LIMIT_PCT
            _start_bal = getattr(self, "_daily_start_balance", 0.0)
            _daily_limit = round(_start_bal * REAL_DAILY_LIMIT_PCT, 2) if _start_bal else 0.0
            if _daily_limit > 0:
                _dp = getattr(self, "_daily_profit", 0.0)
                if _dp >= _daily_limit:
                    print(f"  [RISK] STOP — profit target hari ini tercapai "
                          f"${_dp:.2f} >= ${_daily_limit:.2f} ({REAL_DAILY_LIMIT_PCT*100:.0f}%)")
                    return False
        except Exception:
            pass


        # ── Live open position check (langsung dari MT5, bukan cache) ─────
        from config import MAX_OPEN_POSITIONS
        live_positions = self.mt5.get_positions(self.symbol)
        n_open = len(live_positions) if live_positions else 0
        if n_open >= MAX_OPEN_POSITIONS:
            print(f"  [GUARD] {n_open} posisi masih terbuka "
                  f"(max {MAX_OPEN_POSITIONS}) — tidak buka baru")
            return False

        if self.strict_mode:
            # Sinyal sudah lolos decision engine — trust langsung
            all_pos = self.mt5.get_all_positions(self.symbol)
            if all_pos:
                print(f"  [MICRO] Sudah ada {len(all_pos)} posisi aktif — tunggu tutup dulu")
                return False
            return True

        if self.real_mode:
            from config import (REAL_ML_CONF, REAL_MAX_STACK,
                                REAL_MAX_FLOATING_USD, REAL_SL_COOLDOWN)
            import datetime as _dt

            # ── Reset daily counters jika hari baru ──────────────────────────
            today = _dt.date.today()
            if not hasattr(self, "_daily_date") or self._daily_date != today:
                self._daily_date   = today
                self._daily_loss   = 0.0
                self._daily_profit = 0.0
                # Catat balance awal hari
                try:
                    _bal = self.mt5.get_balance() or 0.0
                    self._daily_start_balance = _bal
                except Exception:
                    pass

            # Hitung limit selalu dari config terbaru × balance awal hari
            try:
                from config import REAL_DAILY_LIMIT_PCT
                _start_bal = getattr(self, "_daily_start_balance", 0.0)
                if not _start_bal:
                    _start_bal = self.mt5.get_balance() or 0.0
                    self._daily_start_balance = _start_bal
                self._daily_limit = round(_start_bal * REAL_DAILY_LIMIT_PCT, 2)
            except Exception:
                pass

            _daily_limit = getattr(self, "_daily_limit", 0.0)

            # ── Daily profit limit (loss tetap jalan) ────────────────────────
            _dp = getattr(self, "_daily_profit", 0.0)
            if _daily_limit > 0 and _dp >= _daily_limit:
                print(f"  [RISK] STOP — profit target hari ini tercapai "
                      f"${_dp:.2f} >= ${_daily_limit:.2f}")
                return False

            all_pos = self.mt5.get_all_positions(self.symbol)

            # ── Stacking limit ───────────────────────────────────────────────
            same = [p for p in all_pos if p["direction"] == direction]
            if len(same) >= REAL_MAX_STACK:
                print(f"  [REAL] Sudah {len(same)} posisi {direction} — "
                      f"batas {REAL_MAX_STACK}x tercapai")
                return False

            # ── ML check: SKIP jika rule-based sudah generate signal valid ───
            # exec_source = "Rule-Based" → sinyal sudah melewati semua filter
            # rule-based (ADX, momentum, konfirmasi, news driver dll)
            # ML tetap dicek HANYA jika sinyal datang dari ML-only path
            exec_source = signal.get("exec_source", "")
            # Sinyal sudah lolos decision engine (ScalpML/Rule/ML filter di bot.py)
            # tidak perlu cek ML lagi di sini
            if same:
                print(f"  [REAL] Posisi {direction}: {len(same)}/{REAL_MAX_STACK} — pasang lagi")
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

        # Circuit breaker: blok order saat market tutup
        if not self.mt5.is_market_open(self.symbol):
            print(f"  [CIRCUIT] Market {self.symbol} sedang tutup — order dibatalkan")
            return {"success": False, "error": "Market tutup", "skipped": True}

        direction = signal.get("direction", "WAIT")
        sl        = signal.get("sl")
        tp        = signal.get("tp")

        if not self.should_trade(signal, ml_pred or {}):
            return {"success": False, "skipped": True, "reason": "guard (cooldown/limit/posisi)"}

        # Ambil harga sekarang dulu — dipakai untuk lot calc + SL resolve
        si    = self.mt5.get_symbol_info(self.symbol)
        price = si.get("ask" if direction == "BUY" else "bid", 0) if si else 0
        sl    = self._resolve_sl(direction, price, sl, self.fixed_lot or 0.01)

        # Signal strength info untuk dynamic lot
        sig_score  = float(signal.get("score", 0))
        ml_conf    = float(ml_pred.get("confidence", 0)) if ml_pred else 0.0

        # SL enforcement dinonaktifkan — pakai ATR multiplier dari config

        # ── Lot cap berdasarkan win rate 10 trade terakhir ───────────────
        from config import MAX_LOT_SAFE, MAX_LOT_LOSING
        _effective_max_lot = MAX_LOT_SAFE   # default safe cap
        try:
            from data.trade_journal import get_stats
            _st = get_stats(self.symbol)
            _recent_wr = _st.get("win_rate", 50) / 100.0
            _total_cl  = _st.get("total", 0)
            if _total_cl >= 5:
                if _recent_wr < 0.35:
                    _effective_max_lot = MAX_LOT_LOSING
                    print(f"  [GUARD] Win rate {_recent_wr:.0%} < 35% — lot dikunci ke {MAX_LOT_LOSING}")
                elif _recent_wr >= 0.50:
                    _effective_max_lot = MAX_LOT_SAFE * 2   # boleh 2x cap kalau WR bagus
        except Exception:
            pass

        lot = self.fixed_lot
        if self.real_mode:
            from config import (REAL_AUTO_LOT, REAL_AUTO_LOT_MIN,
                                REAL_AUTO_LOT_MAX, REAL_RISK_PCT)
            if REAL_AUTO_LOT:
                _real_max = min(REAL_AUTO_LOT_MAX, _effective_max_lot)
                if sl and price and sl != price:
                    # Step 1: hitung base lot dari risk
                    sl_pips  = abs(price - sl) / (si.get("point", 0.01) * 10)
                    base_lot = self.mt5.calculate_lot(
                        self.symbol, sl_pips, REAL_RISK_PCT
                    )
                    base_lot = max(REAL_AUTO_LOT_MIN, min(_real_max, base_lot))

                    # Step 2: scale up berdasarkan kekuatan sinyal
                    lot = self.mt5.calculate_signal_lot(
                        base_lot,
                        signal_score=sig_score,
                        ml_confidence=ml_conf,
                        symbol_key=self.symbol,
                        min_lot=REAL_AUTO_LOT_MIN,
                        max_lot=_real_max,
                    )
                else:
                    # Tidak ada SL — estimasi SL dari ATR (1x ATR default)
                    atr_val = signal.get("atr", 0)
                    _point  = si.get("point", 0.01) if si else 0.01
                    if atr_val and atr_val > 0:
                        sl_pips_est = max(atr_val / (_point * 10), 5.0)
                        base_lot    = self.mt5.calculate_lot(self.symbol, sl_pips_est, REAL_RISK_PCT)
                        base_lot    = max(REAL_AUTO_LOT_MIN, min(_real_max, base_lot))
                        print(f"  [LOT] ATR-based SL est {sl_pips_est:.1f} pip → lot {base_lot:.2f}")
                    else:
                        base_lot = self.mt5.calculate_auto_lot(REAL_AUTO_LOT_MIN, _real_max)
                    lot = self.mt5.calculate_signal_lot(
                        base_lot,
                        signal_score=sig_score,
                        ml_confidence=ml_conf,
                        symbol_key=self.symbol,
                        min_lot=REAL_AUTO_LOT_MIN,
                        max_lot=_real_max,
                    )

        # ── Safety cap universal (berlaku bahkan untuk force-trade) ────────────
        try:
            from config import FORCE_TRADE_MAX_LOT
            if lot and lot > FORCE_TRADE_MAX_LOT:
                print(f"  [SAFETY] Lot {lot:.2f} > cap {FORCE_TRADE_MAX_LOT} — dipotong")
                lot = FORCE_TRADE_MAX_LOT
        except Exception:
            pass

        n = self.bulk_orders
        if n > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from config import ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP

            # sl_dist dan tp_dist dari sinyal (jarak, bukan harga absolut)
            _sl_dist = signal.get("sl_dist") or abs(price - sl) if sl else None
            _tp_dist = signal.get("tp_dist") or abs(tp - price) if tp else None

            def _place(_):
                # Ambil harga aktual saat order ini dikirim
                _si    = self.mt5.get_symbol_info(self.symbol)
                _price = _si.get("ask" if direction == "BUY" else "bid", price) if _si else price
                _atr   = signal.get("atr", _price * 0.001)

                if _sl_dist:
                    _sl = round(_price - _sl_dist if direction == "BUY" else _price + _sl_dist, 2)
                    _tp = round(_price + _tp_dist if direction == "BUY" else _price - _tp_dist, 2) if _tp_dist else None
                else:
                    _sl_d = float(_atr) * ATR_MULTIPLIER_SL
                    _tp_d = float(_atr) * ATR_MULTIPLIER_TP
                    _sl = round(_price - _sl_d if direction == "BUY" else _price + _sl_d, 2)
                    _tp = round(_price + _tp_d if direction == "BUY" else _price - _tp_d, 2)

                return self.mt5.place_order(self.symbol, direction,
                                            lot=lot, sl=_sl, tp=_tp, silent=True)

            success_tickets = []
            failed          = 0
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(_place, i) for i in range(n)]
                for f in as_completed(futures):
                    res = f.result()
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

        # ── FLIP POSISI — close posisi berlawanan sebelum buka yang baru ──────
        # Kalau ada open SELL dan signal BUY masuk (atau sebaliknya),
        # tutup semua posisi berlawanan dulu, baru buka arah baru.
        _opposite = "SELL" if direction == "BUY" else "BUY"
        _opp_positions = [p for p in self.mt5.get_positions(self.symbol)
                          if p["direction"] == _opposite]
        if _opp_positions:
            print(f"  [FLIP] Signal {direction} — close {len(_opp_positions)} posisi {_opposite} dulu")
            for _pos in _opp_positions:
                _cr = self.mt5.close_position(_pos["ticket"])
                if _cr.get("success"):
                    print(f"  [FLIP] #{_pos['ticket']} {_opposite} ditutup ✓")
                else:
                    print(f"  [FLIP] #{_pos['ticket']} gagal tutup: {_cr.get('error')}")

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

        # Set timeframe agar log_entry bisa catat TF yang benar
        self.mt5._journal_tf = getattr(self, "_timeframe", "")
        result = self.mt5.place_order(
            symbol_key=self.symbol,
            direction=direction,
            lot=lot,
            sl=sl,
            tp=tp,
        )
        self.last_sig  = direction
        self.last_tick = time.time()

        # Catat waktu trade untuk cooldown + daily counter
        if result.get("success"):
            _now_ts = time.time()
            self._last_trade_time = _now_ts
            self._trades_this_hour.append(_now_ts)
            self._trades_today += 1
            if result.get("ticket"):
                self._known_tickets.add(result["ticket"])

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

    def sync_closed_positions(self) -> list[dict]:
       
        if not self.mt5.connected:
            return []

        current_open = {p["ticket"] for p in self.mt5.get_positions(self.symbol)}

        # Perbarui _known_tickets dengan posisi yang masih buka
        just_opened  = current_open - self._known_tickets
        just_closed  = self._known_tickets - current_open

        # Tambahkan yang baru dibuka ke tracking
        self._known_tickets |= current_open
        # Hapus yang sudah tutup dari tracking
        self._known_tickets -= just_closed

        import datetime as dt
        from data.trade_journal import log_exit, is_open
        try:
            from services.db_logger import close_trade_in_db
        except Exception:
            close_trade_in_db = None

        # ── Stale OPEN records: tickets in journal/DB but not in MT5 anymore ──
        # Catches positions closed in previous bot sessions
        try:
            import pandas as pd
            from data.trade_journal import JOURNAL_PATH
            if os.path.exists(JOURNAL_PATH):
                df_j = pd.read_csv(JOURNAL_PATH, dtype=str)
                open_in_journal = set(
                    int(t) for t in df_j.loc[df_j["result"] == "OPEN", "ticket"].dropna()
                    if str(t).isdigit()
                )
                stale = open_in_journal - current_open
                just_closed |= stale
        except Exception:
            pass

        if not just_closed:
            return []

        closed_deals = []
        try:
            from_date = dt.datetime.now() - dt.timedelta(days=7)
            deals     = mt5.history_deals_get(from_date, dt.datetime.now()) or []
            # Kumpulkan close deal per POSITION ticket (position_id)
            for d in deals:
                if d.entry == 1:   # entry=1 = OUT (close deal)
                    closed_deals.append({
                        "position_id": getattr(d, "position_id", 0),
                        "order":       d.order,
                        "price":       d.price,
                        "profit":      d.profit,
                        "swap":        getattr(d, "swap", 0.0),
                        "fee":         getattr(d, "fee", 0.0),
                        "comment":     getattr(d, "comment", ""),
                    })
        except Exception:
            pass

        # Map POSITION ticket → close deal (ambil yang paling akhir)
        # Pakai position_id bukan order — position_id cocok dengan tiket posisi yang kita track
        close_map: dict[int, dict] = {}
        for d in closed_deals:
            pid = int(d["position_id"])
            close_map[pid] = d   # overwrite = ambil deal terbaru jika ada 2+

        results = []
        for ticket in just_closed:
            deal = close_map.get(int(ticket))

            if deal is not None:
                price  = float(deal.get("price", 0.0))
                # profit = P&L + swap + fee (net P&L sebenarnya)
                pnl    = float(deal.get("profit", 0.0)) \
                       + float(deal.get("swap", 0.0)) \
                       + float(deal.get("fee", 0.0))
                note   = deal.get("comment", "") or ("sl" if pnl < 0 else "tp")
            else:
                # Deal tidak ditemukan di history — fallback: ambil dari MT5 order history
                price, pnl, note = 0.0, 0.0, "auto-close"
                try:
                    orders = mt5.history_orders_get(
                        position=int(ticket)) or []
                    if orders:
                        last_o = orders[-1]
                        price  = float(getattr(last_o, "price_current", 0.0))
                except Exception:
                    pass

            result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "MANUAL")

            # Loss Control System — dinonaktifkan

            # ── Catat SL time untuk cooldown ─────────────────────────────────
            is_sl_hit = (pnl < 0 and ("sl" in note.lower() or note == "auto-close"))
            if is_sl_hit:
                self._last_sl_time  = time.time()
                self._last_trade_time = time.time()   # reset trade cooldown juga

            # Update daily net profit tracker (WIN - LOSS)
            if not hasattr(self, "_daily_profit"):
                self._daily_profit = 0.0
            if not hasattr(self, "_daily_loss"):
                self._daily_loss = 0.0
            self._daily_profit += pnl          # net: bisa naik (WIN) atau turun (LOSS)
            if pnl < 0:
                self._daily_loss += abs(pnl)   # keep loss tracker untuk referensi

            # Update CSV journal
            try:
                if is_open(ticket):
                    log_exit(self.symbol, self._timeframe, ticket, price, pnl, note=note)
            except Exception:
                pass

            # Update DB + catat ORDER_CLOSE ke tx_log + update signal pnl
            try:
                if close_trade_in_db:
                    close_trade_in_db(
                        ticket, price, pnl, result, note,
                        symbol=self.symbol,
                        timeframe=self._timeframe,
                    )
            except Exception:
                pass

            GREEN = "\033[92m"
            RED   = "\033[91m"
            BOLD  = "\033[1m"
            RESET = "\033[0m"
            rc    = GREEN if result == "WIN" else RED
            print(f"  [Journal] #{ticket} {self.symbol} -> "
                  f"{rc}{BOLD}{result}{RESET}  P&L: {rc}${pnl:+.2f}{RESET}  "
                  f"@ {price:.5f}")

            # ── Adaptive Learning: belajar dari hasil trade ini ──────────────
            try:
                from ai.adaptive import get_learner
                learner = get_learner()

                # Ambil metadata trade dari journal (source, score, direction)
                src_str   = ""
                sig_score = 0.0
                direction = ""
                ind_snap  = {}
                try:
                    import pandas as _pd
                    from data.trade_journal import JOURNAL_PATH
                    if os.path.exists(JOURNAL_PATH):
                        _df_j = _pd.read_csv(JOURNAL_PATH, dtype=str)
                        _row  = _df_j[_df_j["ticket"] == str(ticket)]
                        if not _row.empty:
                            r = _row.iloc[0]
                            direction = str(r.get("direction", ""))
                            src_str   = str(r.get("source", ""))
                            try:
                                sig_score = float(r.get("score", 0) or 0)
                            except Exception:
                                sig_score = 0.0

                        # Ambil snapshot indikator dari candle log
                        from data.candle_log import LOG_DIR
                        cl_path = os.path.join(
                            LOG_DIR, f"candles_{self.symbol}_{getattr(self, '_timeframe', '5m')}.csv")
                        if os.path.exists(cl_path):
                            _cl = _pd.read_csv(cl_path, dtype=str)
                            entry_time = str(r.get("entry_time", ""))[:16]
                            _cl_row = _cl[_cl["time"].str[:16] == entry_time]
                            if not _cl_row.empty:
                                cr = _cl_row.iloc[0]
                                # Buat snapshot boolean: apakah indikator bullish saat entry?
                                try:
                                    rsi_v = float(cr.get("rsi", 50) or 50)
                                    ind_snap["rsi"] = rsi_v > 50
                                except Exception: pass
                                try:
                                    hist_v = float(cr.get("histogram", 0) or 0)
                                    ind_snap["macd"] = hist_v > 0
                                except Exception: pass
                                try:
                                    sig_dir = str(cr.get("signal", ""))
                                    ind_snap["signal_buy"] = sig_dir == "BUY"
                                except Exception: pass
                                try:
                                    td = str(cr.get("tick_dir", "") or "")
                                    if td:
                                        ind_snap["tick"] = td == "BUY"
                                except Exception: pass
                except Exception:
                    pass

                learner.record_trade_outcome(
                    ticket       = int(ticket),
                    result       = result,
                    pnl          = pnl,
                    direction    = direction or "BUY",
                    source       = src_str or "#unknown",
                    signal_score = sig_score,
                    indicator_snapshot = ind_snap if ind_snap else None,
                )
            except Exception:
                pass

            results.append({"ticket": ticket, "result": result, "pnl": pnl,
                             "price": price, "note": note})

        return results

    def manage_positions(self, signal: dict, df=None) -> None:
        direction = signal.get("direction", "WAIT")

        # Reversal close dinonaktifkan — biarkan SL/TP yang menutup posisi
        # for pos in self.mt5.get_positions(self.symbol):
        #     if (pos["direction"] == "BUY"  and direction == "SELL") or \
        #        (pos["direction"] == "SELL" and direction == "BUY"):
        #         profit = pos.get("profit", 0) or 0
        #         if profit > 0:
        #             self.mt5.close_position(pos["ticket"])

        # Auto-management dinonaktifkan — biarkan harga nyentuh SL/TP sendiri
        # self.check_abnormal_movement(df)
        # self.check_sl_risk()
        # self.check_partial_tp()
        # self.check_breakeven()
        # self.check_profit_lock()
        # self.sync_manual_positions(signal)
        # if self.trailing_pips > 0:
        #     all_pos = self.mt5.get_all_positions(self.symbol)
        #     if all_pos:
        #         self.mt5.update_trailing_stop(self.symbol, self.trailing_pips,
        #                                       include_manual=True)

    def check_profit_lock(self) -> None:
        """
        Lock profit: geser SL ke entry+buffer saat floating profit >= PROFIT_LOCK_PCT % dari TP.
        Upgrade 7 — Trade Management Intelligence.
        Contoh: TP = entry + 100 pips, PROFIT_LOCK_PCT=0.5 → lock saat profit >= 50 pips.
        """
        try:
            from config import PROFIT_LOCK_PCT, BREAKEVEN_BUFFER
            if not PROFIT_LOCK_PCT or not MT5_AVAILABLE:
                return
            positions = self.mt5.get_positions(self.symbol)
            if not positions:
                return
            tick = mt5.symbol_info_tick(
                self.symbol if not self.symbol.endswith("m") else self.symbol)
            if not tick:
                return
            bid, ask = tick.bid, tick.ask
            for pos in positions:
                sl  = pos.get("sl", 0) or 0
                tp  = pos.get("tp", 0) or 0
                ep  = pos.get("entry_price", 0) or 0
                drn = pos.get("direction", "")
                tkt = pos.get("ticket", 0)
                if not (tp and ep):
                    continue
                if drn == "BUY":
                    cur     = bid
                    tp_dist = tp - ep
                    cur_prft = cur - ep
                    lock_sl  = ep + BREAKEVEN_BUFFER
                    already_locked = sl >= lock_sl - 0.001
                else:
                    cur     = ask
                    tp_dist = ep - tp
                    cur_prft = ep - cur
                    lock_sl  = ep - BREAKEVEN_BUFFER
                    already_locked = (sl != 0 and sl <= lock_sl + 0.001)
                if tp_dist <= 0 or cur_prft <= 0 or already_locked:
                    continue
                pct = cur_prft / tp_dist
                if pct >= PROFIT_LOCK_PCT:
                    self.mt5.modify_position(tkt, sl=round(lock_sl, 5), tp=tp)
                    print(f"  [ProfitLock] #{tkt} {drn}: SL -> {lock_sl:.5f} "
                          f"({pct:.0%} of TP reached)")
        except Exception:
            pass
