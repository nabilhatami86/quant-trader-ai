"""
Signal Analysis Logger
Menyimpan setiap sinyal ML beserta analisis candle ke logs/signal_analysis.jsonl
Saat trade tutup, outcome (WIN/LOSS/pnl) di-update ke baris yang cocok.
"""

import os
import json
from datetime import datetime
from pathlib import Path

# app/services/ai/ -> app/services/ -> app/ -> root
LOG_PATH = Path(__file__).parent.parent.parent.parent / 'logs' / 'signal_analysis.jsonl'


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_signal(predict_result: dict, symbol: str = 'XAUUSD') -> str:
    """
    Dipanggil setiap kali bot mendapat sinyal dari ScalpingPredictor.
    Simpan ke signal_analysis.jsonl, return signal_id.

    signal_id = "XAUUSD_20260408_183512"
    """
    sig_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ana = predict_result.get('analysis', {})
    record = {
        'signal_id'      : sig_id,
        'time'           : _now(),
        'symbol'         : symbol,

        # Keputusan ML
        'direction'      : predict_result.get('direction', 'WAIT'),
        'prob_buy'       : predict_result.get('prob_buy', 0.0),
        'prob_sell'      : predict_result.get('prob_sell', 0.0),
        'confidence'     : predict_result.get('confidence', ''),
        'close'          : predict_result.get('close', 0.0),
        'sl'             : predict_result.get('sl', 0.0),
        'tp'             : predict_result.get('tp', 0.0),
        'rr'             : predict_result.get('rr', 0.0),
        'atr'            : predict_result.get('atr', 0.0),

        # Analisis candle terstruktur
        'momentum_8c_up' : ana.get('momentum_8c_up'),
        'momentum_8c_dn' : ana.get('momentum_8c_dn'),
        'momentum_8c'    : ana.get('momentum_8c'),
        'momentum_3c'    : ana.get('momentum_3c'),
        'last_candle_dir': ana.get('last_candle_dir'),
        'last_body_pct'  : ana.get('last_body_pct'),
        'last_candle_shape': ana.get('last_candle_shape'),
        'price_pos_pct'  : ana.get('price_pos_pct'),
        'price_pos_label': ana.get('price_pos_label'),
        'atr_rank50'     : ana.get('atr_rank50'),
        'vol_regime'     : ana.get('vol_regime'),
        'consec_streak'  : ana.get('consec_streak', 0),
        'consec_dir'     : ana.get('consec_dir', ''),
        'tp_score'       : ana.get('tp_score'),
        'tp_verdict'     : ana.get('tp_verdict', ''),
        'alignment'      : ana.get('alignment', ''),

        # Notes teks (untuk audit)
        'notes'          : predict_result.get('signal_notes', []),
        'warnings'       : predict_result.get('signal_warnings', []),

        # Outcome — diisi saat trade tutup
        'trade_opened'   : False,
        'ticket'         : None,
        'outcome'        : None,   # WIN / LOSS / MANUAL
        'pnl'            : None,
        'outcome_time'   : None,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return sig_id


def mark_trade_opened(signal_id: str, ticket: int):
    """Tandai bahwa sinyal ini menghasilkan trade (ticket dari MT5)."""
    _update_last_matching(signal_id, {'trade_opened': True, 'ticket': ticket})


def update_outcome(ticket: int, outcome: str, pnl: float):
    """
    Dipanggil dari adaptive.py saat trade tutup.
    Cari entri dengan ticket yang sama, update outcome & pnl.
    """
    if not LOG_PATH.exists():
        return

    lines = LOG_PATH.read_text(encoding='utf-8').splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get('ticket') == ticket and rec.get('outcome') is None:
                rec['outcome']      = outcome
                rec['pnl']          = pnl
                rec['outcome_time'] = _now()
                updated = True
            new_lines.append(json.dumps(rec, ensure_ascii=False))
        except Exception:
            new_lines.append(line)

    if updated:
        LOG_PATH.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')


def _update_last_matching(signal_id: str, fields: dict):
    """Update fields pada record dengan signal_id yang cocok (dari belakang)."""
    if not LOG_PATH.exists():
        return

    lines = LOG_PATH.read_text(encoding='utf-8').splitlines()
    new_lines = []
    done = False
    for line in reversed(lines):
        if not line.strip():
            continue
        if not done:
            try:
                rec = json.loads(line)
                if rec.get('signal_id') == signal_id:
                    rec.update(fields)
                    done = True
                new_lines.append(json.dumps(rec, ensure_ascii=False))
                continue
            except Exception:
                pass
        new_lines.append(line)

    LOG_PATH.write_text('\n'.join(reversed(new_lines)) + '\n', encoding='utf-8')
