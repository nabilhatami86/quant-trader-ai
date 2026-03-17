"""
News Filter - Forex Economic Calendar & Sentiment
Sumber:
  1. ForexFactory calendar (scraping)
  2. NewsAPI (jika ada API key)
  3. Yahoo Finance News
"""
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Opsional: Set API key di sini atau di config
NEWSAPI_KEY = ""    # Daftar gratis di newsapi.org


# ─── IMPACT LEVEL ─────────────────────────────────────────
IMPACT_COLOR = {
    "HIGH":   "RED",
    "MEDIUM": "YELLOW",
    "LOW":    "GREEN",
    "NONE":   "GRAY",
}

# Kata kunci berita yang berdampak pada EURUSD & GOLD
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


class NewsFilter:
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol  = symbol
        self.news    = []
        self.impact  = "NONE"
        self.summary = ""
        self.keywords = KEYWORDS_GOLD if "GOLD" in symbol or "XAU" in symbol else KEYWORDS_EURUSD

    # ─── YAHOO FINANCE NEWS ───────────────────────────────────
    def fetch_yahoo_news(self) -> list:
        """Ambil berita dari Yahoo Finance"""
        ticker_map = {
            "EURUSD": "EURUSD=X",
            "GOLD":   "GC=F",
            "XAUUSD": "GLD",
        }
        ticker_str = ticker_map.get(self.symbol, "EURUSD=X")

        try:
            t     = yf.Ticker(ticker_str)
            news  = t.news or []
            items = []
            for n in news[:10]:
                ct = n.get("content", {})
                title    = ct.get("title", n.get("title", ""))
                pub_date = ct.get("pubDate", "")
                provider = ct.get("provider", {})
                source   = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
                url      = ct.get("canonicalUrl", {})
                link     = url.get("url", "") if isinstance(url, dict) else str(url)

                if not title:
                    continue

                # Deteksi relevance & impact
                text_lower  = title.lower()
                is_relevant = any(k.lower() in text_lower for k in self.keywords)
                if not is_relevant:
                    continue

                impact = self._estimate_impact(title)
                items.append({
                    "title":     title,
                    "source":    source,
                    "published": pub_date,
                    "impact":    impact,
                    "url":       link,
                    "source_api":"Yahoo Finance",
                })
            return items
        except Exception as e:
            return []

    # ─── NEWSAPI ──────────────────────────────────────────────
    def fetch_newsapi(self) -> list:
        """Ambil berita dari NewsAPI (butuh API key gratis)"""
        if not NEWSAPI_KEY:
            return []
        query = "EUR USD forex" if "EUR" in self.symbol else "gold XAU commodity"
        url   = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&language=en&sortBy=publishedAt"
            f"&pageSize=10&apiKey={NEWSAPI_KEY}"
        )
        try:
            r     = requests.get(url, timeout=5)
            data  = r.json()
            items = []
            for a in data.get("articles", [])[:10]:
                title  = a.get("title", "")
                if not title or "[Removed]" in title:
                    continue
                impact = self._estimate_impact(title)
                items.append({
                    "title":     title,
                    "source":    a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", ""),
                    "impact":    impact,
                    "url":       a.get("url", ""),
                    "source_api":"NewsAPI",
                })
            return items
        except Exception:
            return []

    # ─── ECONOMIC CALENDAR (ForexFactory-style) ───────────────
    def fetch_economic_calendar(self) -> list:
        """
        Cek event ekonomi high-impact hari ini.
        Menggunakan API publik dari freeforexapi.com
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        url   = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            r    = requests.get(url, timeout=5)
            data = r.json()
            events = []
            for e in data:
                date_str = e.get("date", "")
                if today not in date_str:
                    continue
                title    = e.get("title", "")
                country  = e.get("country", "")
                impact   = e.get("impact", "Low")
                time_str = e.get("time", "")
                forecast = e.get("forecast", "")
                prev     = e.get("previous", "")

                # Filter relevan
                relevant_countries = ["USD", "EUR"] if "EUR" in self.symbol else ["USD"]
                if country not in relevant_countries:
                    continue

                events.append({
                    "title":     title,
                    "country":   country,
                    "time":      time_str,
                    "impact":    impact.upper(),
                    "forecast":  forecast,
                    "previous":  prev,
                    "source_api":"ForexFactory Calendar",
                })
            return events
        except Exception:
            return []

    # ─── IMPACT ESTIMATOR ─────────────────────────────────────
    def _estimate_impact(self, text: str) -> str:
        """Estimasi dampak berita berdasarkan kata kunci"""
        text = text.lower()
        high_kw = [
            "rate decision", "interest rate", "nfp", "non-farm", "cpi", "inflation",
            "fomc", "ecb", "federal reserve", "gdp", "emergency", "crisis", "war",
            "recession", "default", "sanctions", "collapse",
        ]
        medium_kw = [
            "pmi", "retail sales", "employment", "unemployment", "trade balance",
            "housing", "consumer confidence", "manufacturing",
        ]
        if any(k in text for k in high_kw):
            return "HIGH"
        elif any(k in text for k in medium_kw):
            return "MEDIUM"
        return "LOW"

    # ─── OVERALL SENTIMENT ────────────────────────────────────
    def get_sentiment(self) -> dict:
        """
        Gabungkan semua sumber berita, return sentiment + risk assessment.
        """
        yahoo_news = self.fetch_yahoo_news()
        newsapi    = self.fetch_newsapi()
        calendar   = self.fetch_economic_calendar()

        all_news = yahoo_news + newsapi

        high_count   = sum(1 for n in all_news if n.get("impact") == "HIGH")
        medium_count = sum(1 for n in all_news if n.get("impact") == "MEDIUM")
        cal_high     = [e for e in calendar if e.get("impact") == "HIGH"]

        # Tentukan overall risk
        if high_count >= 2 or cal_high:
            risk_level = "HIGH"
            advice     = "AVOID TRADING - Berita high impact sedang aktif, volatilitas ekstrem"
        elif high_count == 1 or medium_count >= 3:
            risk_level = "MEDIUM"
            advice     = "HATI-HATI - Gunakan SL lebih ketat dari biasa"
        else:
            risk_level = "LOW"
            advice     = "NORMAL - Kondisi aman untuk trading"

        # Top headlines
        headlines = []
        for n in sorted(all_news, key=lambda x: x.get("impact", "LOW"),
                        reverse=True)[:5]:
            headlines.append({
                "title":  n["title"][:80] + ("..." if len(n["title"]) > 80 else ""),
                "impact": n.get("impact", "LOW"),
                "source": n.get("source", ""),
            })

        self.news    = all_news
        self.impact  = risk_level
        self.summary = advice

        return {
            "symbol":       self.symbol,
            "risk_level":   risk_level,
            "advice":       advice,
            "high_news":    high_count,
            "medium_news":  medium_count,
            "total_news":   len(all_news),
            "calendar":     calendar,
            "cal_high":     cal_high,
            "headlines":    headlines,
            "fetched_at":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

    # ─── PRINT ────────────────────────────────────────────────
    def print_news_report(self, sentiment: dict) -> None:
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"

        risk    = sentiment["risk_level"]
        r_color = RED if risk == "HIGH" else YELLOW if risk == "MEDIUM" else GREEN
        sep     = "=" * 60

        print(f"\n{BOLD}{sep}{RESET}")
        print(f"  {BOLD}NEWS & ECONOMIC CALENDAR - {sentiment['symbol']}{RESET}")
        print(f"  {sentiment['fetched_at']}")
        print(sep)
        print(f"  Risk Level  : {BOLD}{r_color}{risk}{RESET}")
        print(f"  Advice      : {r_color}{sentiment['advice']}{RESET}")
        print(f"  News Found  : {sentiment['total_news']} ({sentiment['high_news']} HIGH, {sentiment['medium_news']} MEDIUM)")
        print(sep)

        # Economic Calendar
        if sentiment["calendar"]:
            print(f"  {BOLD}Economic Events Today:{RESET}")
            for e in sentiment["calendar"][:5]:
                imp_c = RED if e["impact"] == "HIGH" else YELLOW if e["impact"] == "MEDIUM" else GREEN
                print(f"    {imp_c}[{e['impact']:6}]{RESET} {e['time']:8} {e['country']} - {e['title']}")
                if e.get("forecast"):
                    print(f"             Forecast: {e['forecast']}  Prev: {e.get('previous', 'N/A')}")
        else:
            print(f"  {GREEN}No high-impact events today from calendar{RESET}")

        # Headlines
        if sentiment["headlines"]:
            print(f"\n  {BOLD}Top Headlines:{RESET}")
            for h in sentiment["headlines"]:
                imp_c = RED if h["impact"] == "HIGH" else YELLOW if h["impact"] == "MEDIUM" else GREEN
                print(f"    {imp_c}[{h['impact']:6}]{RESET} {h['title']}")
                if h.get("source"):
                    print(f"             Source: {h['source']}")
        else:
            print(f"  {GREEN}No relevant news found{RESET}")

        print(sep)
