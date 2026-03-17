"""
MetaTrader 5 Connector
======================
Menghubungkan Trading Bot ke MetaTrader 5 secara langsung.
Fitur:
  - Connect/disconnect ke MT5
  - Place order BUY/SELL otomatis
  - Set Stop Loss & Take Profit
  - Trailing Stop
  - Get posisi aktif
  - Close posisi
  - Get account info
  - Sinkronisasi sinyal dari bot
"""

import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ─── KONFIGURASI MT5 ──────────────────────────────────────
MT5_CONFIG = {
    # Ganti dengan path MT5 Anda (opsional, auto-detect jika kosong)
    "path":     "",                    # contoh: "C:/Program Files/MetaTrader 5/terminal64.exe"
    "server":   "",                    # nama server broker (kosong = default)
    "login":    0,                     # nomor akun MT5 (isi dengan akun Anda)
    "password": "",                    # password akun MT5
    "timeout":  10000,                 # timeout koneksi (ms)
}

# ─── SYMBOL MAPPING ───────────────────────────────────────
# Sesuaikan dengan nama symbol di broker Anda
SYMBOL_MAP = {
    "EURUSD": "EURUSD",       # atau "EURUSDm", "EURUSD." tergantung broker
    "GOLD":   "XAUUSD",       # atau "XAUUSDm", "GOLD"
    "XAUUSD": "XAUUSD",
}

# ─── TRADE CONFIG ─────────────────────────────────────────
DEFAULT_LOT      = 0.01     # Ukuran lot default
MAX_LOT          = 1.0      # Maksimum lot per order
SLIPPAGE         = 10       # Slippage maksimum (points)
MAGIC_NUMBER     = 202601   # ID unik bot ini (jangan ubah)
BOT_COMMENT      = "TraderAI-Bot"

# Risk management: % dari balance per trade
RISK_PERCENT     = 1.0      # 1% risiko per trade


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

        # Init MT5
        kwargs = {"timeout": MT5_CONFIG["timeout"]}
        if path:
            kwargs["path"] = path

        if not mt5.initialize(**kwargs):
            self.last_error = f"MT5 initialize failed: {mt5.last_error()}"
            print(f"[ERROR] {self.last_error}")
            return False

        # Login jika ada credentials
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
            # Coba aktifkan symbol
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if not info:
            return {}
        return {
            "symbol":    symbol,
            "bid":       info.bid,
            "ask":       info.ask,
            "spread":    info.spread,
            "digits":    info.digits,
            "point":     info.point,
            "lot_min":   info.volume_min,
            "lot_max":   info.volume_max,
            "lot_step":  info.volume_step,
            "tick_value":info.trade_tick_value,
            "tick_size": info.trade_tick_size,
        }

    def get_current_price(self, symbol_key: str) -> dict:
        si = self.get_symbol_info(symbol_key)
        return {
            "bid": si.get("bid", 0),
            "ask": si.get("ask", 0),
            "spread": si.get("spread", 0),
        }

    # ─── LOT CALCULATOR ───────────────────────────────────────
    def calculate_lot(self, symbol_key: str, sl_pips: float,
                      risk_percent: float = RISK_PERCENT) -> float:
        """Hitung ukuran lot berdasarkan % risiko dan jarak SL"""
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

        lot = risk_usd / (sl_pips * pip_value)
        lot = max(si.get("lot_min", 0.01), min(lot, min(si.get("lot_max", 100), MAX_LOT)))
        # Round ke step
        step = si.get("lot_step", 0.01)
        lot  = round(round(lot / step) * step, 2)
        return lot

    # ─── PLACE ORDER ──────────────────────────────────────────
    def place_order(self, symbol_key: str, direction: str,
                    lot: float = None, sl: float = None, tp: float = None,
                    comment: str = BOT_COMMENT) -> dict:
        """
        Buka order market.
        direction: "BUY" atau "SELL"
        sl, tp: harga absolut (bukan pips)
        """
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung ke MT5"}

        symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
        mt5.symbol_select(symbol, True)
        si = self.get_symbol_info(symbol_key)
        if not si:
            return {"success": False, "error": f"Symbol {symbol} tidak ditemukan"}

        # Harga
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price      = si["ask"]
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price      = si["bid"]

        # Lot otomatis jika tidak ditentukan
        if lot is None:
            sl_pips = abs(price - sl) / si["point"] / 10 if sl else 20
            lot     = self.calculate_lot(symbol_key, sl_pips)

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    lot,
            "type":      order_type,
            "price":     price,
            "sl":        round(sl, si["digits"]) if sl else 0.0,
            "tp":        round(tp, si["digits"]) if tp else 0.0,
            "deviation": SLIPPAGE,
            "magic":     MAGIC_NUMBER,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            return {"success": False, "error": str(mt5.last_error())}

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] Order {direction} {lot} lot {symbol} @ {price:.5f}")
            print(f"     Ticket: #{result.order}  SL: {sl}  TP: {tp}")
            return {
                "success":  True,
                "ticket":   result.order,
                "direction":direction,
                "symbol":   symbol,
                "lot":      lot,
                "price":    price,
                "sl":       sl,
                "tp":       tp,
                "time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            err = f"Order gagal: retcode={result.retcode} ({result.comment})"
            print(f"[ERROR] {err}")
            return {"success": False, "error": err, "retcode": result.retcode}

    # ─── CLOSE POSITION ───────────────────────────────────────
    def close_position(self, ticket: int) -> dict:
        if not self.connected:
            return {"success": False, "error": "Tidak terhubung"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Posisi #{ticket} tidak ditemukan"}

        pos = positions[0]
        symbol = pos.symbol

        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price      = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price      = mt5.symbol_info_tick(symbol).ask

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    pos.volume,
            "type":      order_type,
            "position":  ticket,
            "price":     price,
            "deviation": SLIPPAGE,
            "magic":     MAGIC_NUMBER,
            "comment":   f"Close #{ticket}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] Posisi #{ticket} berhasil ditutup @ {price:.5f}")
            return {"success": True, "ticket": ticket, "close_price": price}
        return {"success": False, "error": str(mt5.last_error())}

    def close_all(self, symbol_key: str = None) -> list:
        """Tutup semua posisi bot (berdasarkan magic number)"""
        symbol = SYMBOL_MAP.get(symbol_key, symbol_key) if symbol_key else None
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        results = []
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    results.append(self.close_position(pos.ticket))
        return results

    # ─── GET POSITIONS ────────────────────────────────────────
    def get_positions(self, symbol_key: str = None) -> list:
        if not self.connected:
            return []
        symbol = SYMBOL_MAP.get(symbol_key, symbol_key) if symbol_key else None
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if not positions:
            return []

        result = []
        for p in positions:
            if p.magic != MAGIC_NUMBER:
                continue
            result.append({
                "ticket":    p.ticket,
                "symbol":    p.symbol,
                "direction": "BUY" if p.type == 0 else "SELL",
                "lot":       p.volume,
                "open_price":p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "profit":    p.profit,
                "time":      datetime.fromtimestamp(p.time).strftime("%Y-%m-%d %H:%M"),
                "comment":   p.comment,
            })
        return result

    def get_history(self, days: int = 7) -> list:
        """Ambil history trade bot"""
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
                "ticket":  d.ticket,
                "symbol":  d.symbol,
                "type":    "BUY" if d.type == 0 else "SELL",
                "lot":     d.volume,
                "price":   d.price,
                "profit":  d.profit,
                "time":    datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M"),
            })
        return result

    # ─── PRINT STATUS ─────────────────────────────────────────
    def print_status(self) -> None:
        if not self.connected:
            print("[!] Tidak terhubung ke MT5")
            return

        self._refresh_account()
        positions = self.get_positions()

        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        sep    = "=" * 55

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  MT5 ACCOUNT STATUS")
        print(sep)
        print(f"  Broker  : {self.account.get('company', 'Unknown')}")
        print(f"  Account : {self.account.get('login', 0)}  ({self.account.get('server', '')})")
        print(f"  Balance : {BOLD}${self.account.get('balance', 0):,.2f}{RESET}")
        print(f"  Equity  : ${self.account.get('equity', 0):,.2f}")
        margin_free = self.account.get('margin_free', 0)
        print(f"  Margin  : ${self.account.get('margin', 0):,.2f}  Free: ${margin_free:,.2f}")
        profit = self.account.get('profit', 0)
        pc = GREEN if profit >= 0 else RED
        print(f"  Float P&L: {pc}{BOLD}${profit:+,.2f}{RESET}")
        print(sep)

        if positions:
            print(f"  {BOLD}Open Positions ({len(positions)}):{RESET}")
            for p in positions:
                pc = GREEN if p["profit"] >= 0 else RED
                dc = GREEN if p["direction"] == "BUY" else RED
                print(f"    #{p['ticket']} {dc}{p['direction']}{RESET} {p['symbol']} "
                      f"{p['lot']}lot @ {p['open_price']:.5f} | "
                      f"P&L: {pc}${p['profit']:+.2f}{RESET}")
        else:
            print(f"  {YELLOW}Tidak ada posisi aktif dari bot ini{RESET}")
        print(sep)


# ─── SIGNAL EXECUTOR ──────────────────────────────────────
class SignalExecutor:
    """
    Menghubungkan sinyal dari Trading Bot ke MT5.
    Otomatis buka/tutup order berdasarkan sinyal.
    """

    def __init__(self, connector: MT5Connector, symbol_key: str):
        self.mt5       = connector
        self.symbol    = symbol_key
        self.last_sig  = None
        self.last_tick = 0

    def should_trade(self, signal: dict, ml_pred: dict,
                     news_risk: str = "LOW") -> bool:
        """Filter: apakah layak masuk trade?"""
        direction = signal.get("direction", "WAIT")

        # Jangan trade jika WAIT
        if direction == "WAIT":
            return False

        # Jangan trade jika ada berita HIGH impact
        if news_risk == "HIGH":
            print(f"[!] Trade dibatalkan - High Impact News aktif")
            return False

        # Jangan trade jika ML tidak confident (jika tersedia)
        if ml_pred and ml_pred.get("direction") not in (direction, "WAIT"):
            print(f"[!] Trade dibatalkan - ML tidak sepakat: Signal={direction}, ML={ml_pred.get('direction')}")
            return False

        # Cek apakah sudah ada posisi aktif
        positions = self.mt5.get_positions(self.symbol)
        if positions:
            existing = positions[0]["direction"]
            if existing == direction:
                print(f"[~] Sudah ada posisi {direction} aktif, skip")
                return False

        return True

    def execute(self, signal: dict, ml_pred: dict = None,
                news_risk: str = "LOW") -> dict:
        """Eksekusi sinyal ke MT5"""
        if not self.mt5.connected:
            return {"success": False, "error": "MT5 tidak terhubung"}

        direction = signal.get("direction", "WAIT")
        sl        = signal.get("sl")
        tp        = signal.get("tp")

        if not self.should_trade(signal, ml_pred or {}, news_risk):
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

    def manage_positions(self, signal: dict) -> None:
        """Tutup posisi jika sinyal berbalik"""
        direction = signal.get("direction", "WAIT")
        positions = self.mt5.get_positions(self.symbol)

        for pos in positions:
            # Jika sinyal berbalik arah, tutup posisi
            if (pos["direction"] == "BUY"  and direction == "SELL") or \
               (pos["direction"] == "SELL" and direction == "BUY"):
                print(f"[~] Sinyal berbalik - menutup posisi #{pos['ticket']}")
                self.mt5.close_position(pos["ticket"])
