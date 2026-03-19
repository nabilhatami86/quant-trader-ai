import os
import json
import requests
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "news_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Berapa hari ke belakang yang diperhitungkan efeknya
NEWS_LOOKBACK_DAYS = 5

KEYWORDS_EURUSD = [
    "EUR", "USD", "ECB", "Fed", "Federal Reserve", "interest rate",
    "inflation", "CPI", "NFP", "Non-Farm", "GDP", "PMI", "FOMC",
    "European Central Bank", "Powell", "Lagarde", "dollar", "euro",
]


KEYWORDS_GOLD = [
    "gold", "XAU", "GLD", "inflation", "Fed", "interest rate",
    "safe haven", "risk off", "geopolitical", "war", "crisis",
    "DXY", "dollar index", "yield", "treasury",
]

# keyword: (score, effect_days) — positif=BULLISH, negatif=BEARISH
DIRECTION_KEYWORDS_GOLD = {
    # BULLISH geopolitik / krisis
    "war":                (+3.0, 5),
    "conflict":           (+2.5, 4),
    "attack":             (+2.5, 4),
    "missile":            (+2.5, 4),
    "airstrike":          (+2.5, 4),
    "nuclear":            (+3.0, 5),
    "geopolitical":       (+2.5, 4),
    "sanctions":          (+2.0, 4),
    "invasion":           (+3.0, 5),
    "crisis":             (+2.0, 3),
    "banking crisis":     (+2.5, 3),
    "bank collapse":      (+2.5, 3),
    "debt ceiling":       (+2.0, 3),
    "recession":          (+2.0, 3),
    "safe haven":         (+2.0, 2),
    "risk aversion":      (+2.0, 2),
    "risk off":           (+2.0, 2),
    "uncertainty":        (+1.5, 2),

    # BULLISH kebijakan moneter
    "rate cut":           (+3.0, 2),
    "cuts rate":          (+3.0, 2),
    "rate cuts":          (+3.0, 2),
    "dovish":             (+2.5, 2),
    "pause hike":         (+2.0, 2),

    # BULLISH data ekonomi
    "inflation rise":     (+2.0, 2),
    "inflation surge":    (+2.0, 2),
    "inflation high":     (+2.0, 2),
    "hot inflation":      (+2.0, 2),
    "cpi beat":           (+2.0, 2),
    "cpi higher":         (+2.0, 2),
    "dollar falls":       (+2.0, 1),
    "dollar weakens":     (+2.0, 1),
    "dollar weakness":    (+2.0, 1),
    "dxy drops":          (+2.0, 1),
    "dxy falls":          (+2.0, 1),
    "yield falls":        (+1.5, 1),
    "yields fall":        (+1.5, 1),
    "yield drop":         (+1.5, 1),
    "slowdown":           (+1.5, 2),
    "weak jobs":          (+1.5, 1),
    "jobs miss":          (+1.5, 1),
    "nfp miss":           (+2.0, 2),
    "below forecast":     (+1.0, 1),
    "gold rally":         (+1.5, 1),
    "gold rises":         (+1.5, 1),
    "buy gold":           (+1.0, 1),

    # BEARISH geopolitik
    "ceasefire":          (-2.0, 3),
    "peace deal":         (-2.0, 3),
    "peace agreement":    (-2.0, 3),
    "de-escalation":      (-1.5, 2),
    "troops withdraw":    (-1.5, 2),

    # BEARISH kebijakan moneter
    "rate hike":          (-3.0, 2),
    "hikes rate":         (-3.0, 2),
    "rate hikes":         (-3.0, 2),
    "hawkish":            (-2.5, 2),
    "higher for longer":  (-2.0, 2),
    "tightening":         (-2.0, 2),

    # BEARISH data ekonomi
    "dollar rises":       (-2.0, 1),
    "dollar strengthens": (-2.0, 1),
    "dollar strength":    (-2.0, 1),
    "dxy rises":          (-2.0, 1),
    "dxy up":             (-2.0, 1),
    "yield rises":        (-2.0, 1),
    "yields rise":        (-2.0, 1),
    "yield surge":        (-2.0, 1),
    "risk on":            (-2.0, 1),
    "stock rally":        (-1.0, 1),
    "strong jobs":        (-1.5, 1),
    "nfp beat":           (-2.0, 2),
    "jobs beat":          (-2.0, 1),
    "above forecast":     (-1.0, 1),
    "inflation cool":     (-2.0, 2),
    "inflation falls":    (-2.0, 2),
    "cpi miss":           (-2.0, 2),
    "cpi cool":           (-2.0, 2),
    "gold falls":         (-1.5, 1),
    "gold drops":         (-1.5, 1),
    "sell gold":          (-1.0, 1),
    "selloff":            (-1.0, 1),
}

DIRECTION_KEYWORDS_EURUSD = {
    "ecb hike":           (+3.0, 2),
    "ecb raises":         (+3.0, 2),
    "ecb hawkish":        (+2.5, 2),
    "lagarde hawkish":    (+2.0, 2),
    "eu gdp beat":        (+2.0, 1),
    "eurozone growth":    (+2.0, 2),
    "fed cut":            (+2.0, 2),
    "fed dovish":         (+2.0, 2),
    "dollar falls":       (+2.0, 1),
    "dollar weakens":     (+2.0, 1),
    "weak nfp":           (+2.0, 2),
    "nfp miss":           (+2.0, 2),
    "jobs miss":          (+1.5, 1),
    "fed hike":           (-3.0, 2),
    "fed hawkish":        (-2.5, 2),
    "powell hawkish":     (-2.0, 2),
    "rate hike":          (-2.0, 2),
    "nfp beat":           (-2.0, 2),
    "strong dollar":      (-2.0, 1),
    "dollar rises":       (-2.0, 1),
    "ecb dovish":         (-3.0, 2),
    "ecb cut":            (-3.0, 2),
    "eu recession":       (-2.0, 3),
    "eurozone weak":      (-2.0, 2),
}

CALENDAR_IMPACT_GOLD = {
    "Non-Farm Payrolls":      (-2.0, +2.0, 2),
    "CPI":                    (+2.0, -2.0, 2),
    "Core CPI":               (+2.0, -2.0, 2),
    "FOMC":                   (-1.0,  0.0, 2),
    "Interest Rate Decision": (-2.0, +2.0, 2),
    "GDP":                    (-1.0, +1.0, 1),
    "Unemployment Rate":      (+1.0, -1.0, 1),
    "PMI":                    (-1.0, +1.0, 1),
    "Retail Sales":           (-1.0, +1.0, 1),
}


class NewsFilter:
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol   = symbol
        self.news     = []
        self.impact   = "NONE"
        self.summary  = ""
        self.keywords = KEYWORDS_GOLD if "GOLD" in symbol or "XAU" in symbol else KEYWORDS_EURUSD
        self.dir_kw   = DIRECTION_KEYWORDS_GOLD if "GOLD" in symbol or "XAU" in symbol \
                        else DIRECTION_KEYWORDS_EURUSD
        self._cached_sentiment = None

    def _date_cache_path(self, date: datetime) -> str:
        date_str = date.strftime("%Y-%m-%d")
        return os.path.join(CACHE_DIR, f"{self.symbol}_{date_str}.json")

    def _load_day_cache(self, date: datetime) -> dict | None:
        path = self._date_cache_path(date)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_day_cache(self, data: dict, date: datetime) -> None:
        path = self._date_cache_path(date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _today_cache_fresh(self) -> bool:
        """Apakah cache hari ini sudah ada?"""
        return self._load_day_cache(datetime.utcnow()) is not None

    def fetch_yahoo_news(self) -> list:
        ticker_map = {"EURUSD": "EURUSD=X", "GOLD": "GC=F", "XAUUSD": "GLD"}
        ticker_str = ticker_map.get(self.symbol, "EURUSD=X")
        try:
            t    = yf.Ticker(ticker_str)
            news = t.news or []
            items = []
            for n in news[:15]:
                ct       = n.get("content", {})
                title    = ct.get("title", n.get("title", ""))
                pub_date = ct.get("pubDate", "")
                provider = ct.get("provider", {})
                source   = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
                url_obj  = ct.get("canonicalUrl", {})
                link     = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                if not title:
                    continue
                if not any(k.lower() in title.lower() for k in self.keywords):
                    continue
                items.append({
                    "title":      title,
                    "source":     source,
                    "published":  pub_date,
                    "impact":     self._estimate_impact(title),
                    "url":        link,
                    "source_api": "Yahoo Finance",
                    "date":       datetime.utcnow().strftime("%Y-%m-%d"),
                })
            return items
        except Exception:
            return []

    def fetch_newsapi(self) -> list:
        if not NEWSAPI_KEY:
            return []
        query = "EUR USD forex" if "EUR" in self.symbol else "gold XAU commodity"
        url = (f"https://newsapi.org/v2/everything?q={query}&language=en"
               f"&sortBy=publishedAt&pageSize=10&apiKey={NEWSAPI_KEY}")
        try:
            r    = requests.get(url, timeout=5)
            data = r.json()
            items = []
            for a in data.get("articles", [])[:10]:
                title = a.get("title", "")
                if not title or "[Removed]" in title:
                    continue
                items.append({
                    "title":      title,
                    "source":     a.get("source", {}).get("name", ""),
                    "published":  a.get("publishedAt", ""),
                    "impact":     self._estimate_impact(title),
                    "url":        a.get("url", ""),
                    "source_api": "NewsAPI",
                    "date":       datetime.utcnow().strftime("%Y-%m-%d"),
                })
            return items
        except Exception:
            return []

    def fetch_economic_calendar(self) -> list:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        url   = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            r    = requests.get(url, timeout=5)
            data = r.json()
            events = []
            for e in data:
                if today not in e.get("date", ""):
                    continue
                country = e.get("country", "")
                relevant = ["USD", "EUR"] if "EUR" in self.symbol else ["USD"]
                if country not in relevant:
                    continue
                events.append({
                    "title":      e.get("title", ""),
                    "country":    country,
                    "time":       e.get("time", ""),
                    "impact":     e.get("impact", "Low").upper(),
                    "forecast":   e.get("forecast", ""),
                    "previous":   e.get("previous", ""),
                    "actual":     e.get("actual", ""),
                    "date":       today,
                    "source_api": "ForexFactory Calendar",
                })
            return events
        except Exception:
            return []

    def _estimate_impact(self, text: str) -> str:
        text = text.lower()
        high_kw = ["rate decision","interest rate","nfp","non-farm","cpi","inflation",
                   "fomc","ecb","federal reserve","gdp","emergency","crisis","war",
                   "recession","default","sanctions","collapse","nuclear","invasion",
                   "attack","missile","airstrike"]
        medium_kw = ["pmi","retail sales","employment","unemployment","trade balance",
                     "housing","consumer confidence","manufacturing","geopolitical",
                     "conflict","ceasefire"]
        if any(k in text for k in high_kw):
            return "HIGH"
        elif any(k in text for k in medium_kw):
            return "MEDIUM"
        return "LOW"

    def _decay_factor(self, hours_ago: float, effect_days: int) -> float:
        """
        Hitung faktor peluruhan berita.
        Dalam rentang effect_days pertama: 1.0 → 0.1 (linear decay)
        Setelah effect_days: 0 (tidak berpengaruh lagi)
        """
        max_hours = effect_days * 24
        if hours_ago <= 0:
            return 1.0
        if hours_ago >= max_hours:
            return 0.0
        # Linear decay dari 1.0 ke 0.1 dalam rentang effect_days
        ratio = hours_ago / max_hours
        return round(1.0 - (ratio * 0.9), 3)

    def _load_all_recent_news(self) -> list:
        """
        Kumpulkan semua berita dari N hari terakhir (dari file cache harian).
        Setiap berita diberi field 'hours_ago' untuk decay calculation.
        """
        all_items = []
        now = datetime.utcnow()

        for day_offset in range(NEWS_LOOKBACK_DAYS):
            date = now - timedelta(days=day_offset)
            cached = self._load_day_cache(date)
            if not cached:
                continue

            # Titik waktu berita: asumsikan jam 12:00 UTC hari itu
            news_time = date.replace(hour=12, minute=0, second=0, microsecond=0)
            hours_ago = (now - news_time).total_seconds() / 3600

            for item in cached.get("news", []):
                enriched = dict(item)
                enriched["hours_ago"]   = round(hours_ago, 1)
                enriched["day_offset"]  = day_offset
                enriched["news_date"]   = date.strftime("%Y-%m-%d")
                all_items.append(enriched)

            # Calendar events dari hari itu juga
            for event in cached.get("calendar", []):
                enriched = dict(event)
                enriched["hours_ago"]  = round(hours_ago, 1)
                enriched["day_offset"] = day_offset
                enriched["news_date"]  = date.strftime("%Y-%m-%d")
                enriched["_is_event"]  = True
                all_items.append(enriched)

        return all_items

    def get_direction_bias(self,
                           news_list: list | None = None,
                           calendar: list | None  = None,
                           use_history: bool = True) -> dict:
        """
        Analisis berita (termasuk hari-hari sebelumnya) → prediksi arah candle.

        Berita lama masih dihitung tapi dengan bobot lebih kecil (decay).
        Perang/krisis/geopolitik punya efek jauh lebih panjang dari data ekonomi.

        Returns:
            {
              "bias":        "BULLISH"|"BEARISH"|"NEUTRAL",
              "score":       float (-10 .. +10),
              "confidence":  "HIGH"|"MEDIUM"|"LOW",
              "active_events": [{"title", "date", "hours_ago", "contribution", "direction"}],
              "reasons":     [str],
            }
        """
        total_score  = 0.0
        active_events = []
        reasons      = []

        if use_history:
            all_items = self._load_all_recent_news()
        else:
            # Hanya berita hari ini (mode fallback)
            all_items = []
            now = datetime.utcnow()
            for item in (news_list or []):
                enriched = dict(item)
                enriched["hours_ago"]  = 0
                enriched["day_offset"] = 0
                all_items.append(enriched)
            for event in (calendar or []):
                enriched = dict(event)
                enriched["hours_ago"]  = 0
                enriched["day_offset"] = 0
                enriched["_is_event"]  = True
                all_items.append(enriched)

        if news_list:
            for item in news_list:
                enriched = dict(item)
                enriched["hours_ago"]  = 0.5   # baru saja
                enriched["day_offset"] = 0
                all_items.append(enriched)
        if calendar:
            for ev in calendar:
                enriched = dict(ev)
                enriched["hours_ago"]  = 0.5
                enriched["day_offset"] = 0
                enriched["_is_event"]  = True
                all_items.append(enriched)

        seen_titles = set()   # hindari double-count

        for item in all_items:
            title     = item.get("title", "")
            title_key = title[:50].lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            impact    = item.get("impact", "LOW")
            hours_ago = item.get("hours_ago", 0)
            news_date = item.get("news_date", "hari ini")
            is_event  = item.get("_is_event", False)
            day_off   = item.get("day_offset", 0)
            imp_weight = 1.5 if impact == "HIGH" else 1.0 if impact == "MEDIUM" else 0.6

            if is_event:
                actual   = item.get("actual", "")
                forecast = item.get("forecast", "")
                ev_title = item.get("title", "")
                for ev_name, (beat_sc, miss_sc, eff_days) in CALENDAR_IMPACT_GOLD.items():
                    if ev_name.lower() in ev_title.lower():
                        decay = self._decay_factor(hours_ago, eff_days)
                        if decay == 0:
                            break
                        if actual and forecast:
                            try:
                                av = float(actual.replace("K","000").replace("M","000000").replace("%","").strip())
                                fv = float(forecast.replace("K","000").replace("M","000000").replace("%","").strip())
                                raw_sc = beat_sc if av > fv else miss_sc
                                label  = "BEAT" if av > fv else "MISS"
                            except ValueError:
                                raw_sc, label = 0, "?"
                        elif impact == "HIGH" and not actual:
                            raw_sc, label = 0, "UPCOMING"
                        else:
                            break

                        contrib = round(raw_sc * decay, 2)
                        total_score += contrib
                        d_word = "BULLISH" if contrib > 0 else "BEARISH" if contrib < 0 else "NEUTRAL"
                        ago_str = f"{day_off}h lalu" if day_off == 0 else f"{day_off} hari lalu"
                        active_events.append({
                            "title":        f"[KALENDER] {ev_title}",
                            "date":         news_date,
                            "hours_ago":    hours_ago,
                            "decay":        decay,
                            "contribution": contrib,
                            "direction":    d_word,
                        })
                        if abs(contrib) >= 0.3:
                            reasons.append(
                                f"[{d_word:7}|{ago_str}|decay:{decay}] "
                                f"{ev_title}: {label} → skor {contrib:+.1f}"
                            )
                        break
            else:
                title_low = title.lower()
                for kw, (kw_score, eff_days) in self.dir_kw.items():
                    if kw in title_low:
                        decay  = self._decay_factor(hours_ago, eff_days)
                        if decay == 0:
                            break
                        contrib = round(kw_score * imp_weight * decay, 2)
                        total_score += contrib
                        d_word = "BULLISH" if contrib > 0 else "BEARISH"
                        day_off = item.get("day_offset", 0)
                        ago_str = "hari ini" if day_off == 0 else f"{day_off} hari lalu"
                        active_events.append({
                            "title":        title,
                            "date":         news_date,
                            "hours_ago":    hours_ago,
                            "decay":        decay,
                            "contribution": contrib,
                            "direction":    d_word,
                            "keyword":      kw,
                        })
                        if abs(contrib) >= 0.5:
                            reasons.append(
                                f"[{d_word:7}|{ago_str}|decay:{decay:.1f}|{impact}] "
                                f"'{kw}' → {title[:60]}..."
                            )
                        break

        clamped = max(-10.0, min(10.0, total_score))

        bias       = "BULLISH" if clamped >= 3 else "BEARISH" if clamped <= -3 else "NEUTRAL"
        abs_score  = abs(clamped)
        confidence = "HIGH" if abs_score >= 6 else "MEDIUM" if abs_score >= 3 else "LOW"

        # Sort event berdasarkan kontribusi terbesar
        active_events.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "bias":          bias,
            "score":         round(clamped, 2),
            "confidence":    confidence,
            "active_events": active_events[:10],
            "reasons":       reasons[:10],
        }

    def get_sentiment(self, use_cache: bool = True) -> dict:
        """
        Fetch berita hari ini (jika belum ada di cache).
        Bias dihitung dari berita hari ini + N hari ke belakang.
        """
        today = datetime.utcnow()

        # Cek cache hari ini
        today_cached = self._load_day_cache(today)
        if use_cache and today_cached:
            # Cache ada — hanya hitung ulang bias (pakai history)
            news_today    = today_cached.get("news", [])
            calendar_today= today_cached.get("calendar", [])
            from_cache    = True
        else:
            # Fetch live
            print(f"[~] Fetching berita hari ini ({self.symbol})...")
            news_today    = self.fetch_yahoo_news() + self.fetch_newsapi()
            calendar_today= self.fetch_economic_calendar()
            from_cache    = False

            # Simpan ke history harian
            self._save_day_cache({
                "date":     today.strftime("%Y-%m-%d"),
                "symbol":   self.symbol,
                "news":     news_today,
                "calendar": calendar_today,
            }, today)

        all_news     = news_today
        high_count   = sum(1 for n in all_news if n.get("impact") == "HIGH")
        medium_count = sum(1 for n in all_news if n.get("impact") == "MEDIUM")
        cal_high     = [e for e in calendar_today if e.get("impact") == "HIGH"]

        if high_count >= 2 or cal_high:
            risk_level = "HIGH"
            advice     = "AVOID TRADING - Berita high impact sedang aktif"
        elif high_count == 1 or medium_count >= 3:
            risk_level = "MEDIUM"
            advice     = "HATI-HATI - Gunakan SL lebih ketat"
        else:
            risk_level = "LOW"
            advice     = "NORMAL - Kondisi aman untuk trading"

        # Bias dengan lag effect (berita hari ini + historis)
        direction_bias = self.get_direction_bias(
            news_list=news_today,
            calendar=calendar_today,
            use_history=True,
        )

        headlines = []
        for n in sorted(all_news, key=lambda x: x.get("impact","LOW"), reverse=True)[:5]:
            headlines.append({
                "title":  n["title"][:80] + ("..." if len(n["title"]) > 80 else ""),
                "impact": n.get("impact","LOW"),
                "source": n.get("source",""),
            })

        self.news   = all_news
        self.impact = risk_level
        self.summary= advice

        result = {
            "symbol":         self.symbol,
            "risk_level":     risk_level,
            "advice":         advice,
            "high_news":      high_count,
            "medium_news":    medium_count,
            "total_news":     len(all_news),
            "calendar":       calendar_today,
            "cal_high":       cal_high,
            "headlines":      headlines,
            "direction_bias": direction_bias,
            "from_cache":     from_cache,
            "fetched_at":     today.strftime("%Y-%m-%d %H:%M UTC"),
        }
        self._cached_sentiment = result
        return result

    def print_news_report(self, sentiment: dict) -> None:
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"
        DIM    = "\033[2m"

        risk    = sentiment["risk_level"]
        r_color = RED if risk == "HIGH" else YELLOW if risk == "MEDIUM" else GREEN
        bias_d  = sentiment.get("direction_bias", {})
        bias    = bias_d.get("bias", "NEUTRAL")
        b_score = bias_d.get("score", 0)
        b_conf  = bias_d.get("confidence", "LOW")
        b_color = GREEN if bias == "BULLISH" else RED if bias == "BEARISH" else YELLOW
        sep     = "=" * 62
        tag     = f"{DIM}[CACHE]{RESET}" if sentiment.get("from_cache") else f"{GREEN}[LIVE]{RESET}"

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  {BOLD}NEWS & ECONOMIC CALENDAR - {sentiment['symbol']}{RESET}")
        print(f"  {sentiment['fetched_at']}  {tag}")
        print(sep)
        print(f"  Risk Level   : {BOLD}{r_color}{risk}{RESET}")
        print(f"  Advice       : {r_color}{sentiment['advice']}{RESET}")
        print(f"  Berita Hari  : {sentiment['total_news']} artikel "
              f"({sentiment['high_news']} HIGH, {sentiment['medium_news']} MEDIUM)")
        print(sep)

        print(f"\n  {BOLD}[PREDIKSI ARAH CANDLE DARI BERITA + HISTORIS]{RESET}")
        print(f"  Bias    : {BOLD}{b_color}{bias}{RESET}  (skor: {b_score:+.1f}/10)")
        print(f"  Konfiden: {b_conf}")
        print()

        events = bias_d.get("active_events", [])
        if events:
            print(f"  {BOLD}Event Aktif yang Masih Memengaruhi Harga:{RESET}")
            for ev in events[:8]:
                d     = ev["direction"]
                ec    = GREEN if d == "BULLISH" else RED if d == "BEARISH" else YELLOW
                decay = ev.get("decay", 1.0)
                cont  = ev.get("contribution", 0)
                date  = ev.get("date", "")
                hours = ev.get("hours_ago", 0)
                ago   = "baru saja" if hours < 2 else \
                        f"{hours:.0f} jam lalu" if hours < 24 else \
                        f"{hours/24:.0f} hari lalu"
                kw    = ev.get("keyword", "")
                kw_str= f" ['{kw}']" if kw else ""
                print(f"    {ec}[{d:7}]{RESET} {cont:+.1f}  "
                      f"{DIM}decay:{decay:.1f} | {ago} ({date}){RESET}")
                print(f"           {ev['title'][:72]}{kw_str}")
        else:
            print(f"  {DIM}Tidak ada event aktif dalam {NEWS_LOOKBACK_DAYS} hari terakhir{RESET}")

        print()
        if sentiment["calendar"]:
            print(f"  {BOLD}Economic Events Hari Ini:{RESET}")
            for e in sentiment["calendar"][:6]:
                ic     = RED if e["impact"] == "HIGH" else YELLOW if e["impact"] == "MEDIUM" else GREEN
                actual = f"  → Actual: {BOLD}{e['actual']}{RESET}" if e.get("actual") else ""
                print(f"    {ic}[{e['impact']:6}]{RESET} {e['time']:8} {e['country']} - {e['title']}")
                if e.get("forecast") or actual:
                    print(f"             Forecast: {e.get('forecast','N/A')}  "
                          f"Prev: {e.get('previous','N/A')}{actual}")
        else:
            print(f"  {GREEN}Tidak ada event high-impact hari ini{RESET}")

        if sentiment["headlines"]:
            print(f"\n  {BOLD}Top Headlines Hari Ini:{RESET}")
            for h in sentiment["headlines"]:
                ic = RED if h["impact"] == "HIGH" else YELLOW if h["impact"] == "MEDIUM" else GREEN
                print(f"    {ic}[{h['impact']:6}]{RESET} {h['title']}")

        print(sep)
