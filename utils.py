"""utils.py — Shared constants, API helpers, demo data, UI components."""
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ── API base ──────────────────────────────────────────────────────────────────
API_BASE = "https://fraud-detection-193665.onrender.com"

# ── Lookup tables ─────────────────────────────────────────────────────────────
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
CHANNELS   = ["Mobile Banking", "Internet Banking", "Branch", "ATM", "USSD", "Agent"]
REGIONS    = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
              "Thika", "Nyeri", "Garissa", "Kisii", "Machakos"]
ACCOUNT_NAMES = [
    "Kamau Enterprises Ltd", "Wanjiku Holdings", "Otieno & Sons",
    "Njeri Traders", "Kipchoge Logistics", "Achieng Retail",
    "Mwangi Digital", "Chebet Imports", "Odhiambo Finance",
    "Nyambura Investments", "Kariuki Motors", "Adhiambo Agency",
    "Gitau Properties", "Auma Consultants", "Mutua Electronics",
    "Waweru Supplies", "Kimani Tech", "Aoko Services",
    "Njoroge & Partners", "Wekesa Holdings",
]
INCIDENT_TYPES = [
    "Single Vehicle Collision", "Multi-Vehicle Collision",
    "Vehicle Theft", "Parked Car", "Other",
]
SEVERITY_OPTS   = ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
AUTO_MAKES      = ["Toyota", "Nissan", "Subaru", "Mitsubishi", "Honda", "Isuzu", "Hyundai", "Ford"]
CHANNELS_CLAIM  = ["BROKER", "DIRECT", "AGENT", "BANCASSURANCE"]

PAGES = {
    "🏠  Executive Dashboard":  "Home",
    "🔍  Claim Prediction":      "Predict",
    "📦  Batch Scoring":         "Batch",
    "📊  Analytics & Trends":    "Analytics",
    "🏦  Account Intelligence":  "Accounts",
    "🚨  Risk & Alert Centre":   "Alerts",
    "⚖️  Model Performance":     "Performance",
    "📋  Transaction Log":       "Log",
    "📑  Management Reports":    "Reports",
    "⚙️  API & Settings":        "API",
}

# ── Colour palette (Power BI inspired) ───────────────────────────────────────
C = {
    "navy":   "#1B3A6B",
    "blue":   "#2B5BA8",
    "red":    "#D13438",
    "green":  "#107C10",
    "amber":  "#F2B705",
    "orange": "#E17B25",
    "grey":   "#605E5C",
    "lgrey":  "#EDEBE9",
    "bg":     "#F3F2F1",
    "white":  "#FFFFFF",
    "dark":   "#252423",
}
COLOR_MAP = {
    "FRAUD":      C["red"],
    "LEGITIMATE": "#2B5BA8",
    "HIGH":       C["red"],
    "MEDIUM":     C["amber"],
    "LOW":        C["green"],
}
CHART_CFG = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#252423", size=11, family="Segoe UI, sans-serif"),
    margin=dict(l=4, r=4, t=32, b=4),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
GRID = dict(gridcolor="#EDEBE9", zerolinecolor="#EDEBE9", linecolor="#E8E6E3")

# ── API helpers ───────────────────────────────────────────────────────────────
def api_health():
    """Returns (ok: bool, model_name: str, latency_ms: int)."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=8)
        if r.status_code == 200:
            d = r.json()
            return True, d.get("model", "unknown"), int(r.elapsed.total_seconds() * 1000)
    except Exception:
        pass
    return False, "offline", 0

def api_predict(payload: dict):
    """POST /predict. Returns (result_dict | None, error_str | None)."""
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=25)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "Cannot reach API — Render free tier may be sleeping. Wait 30 s and retry."
    except requests.Timeout:
        return None, "Request timed out (cold start). Retry in 20 seconds."
    except requests.HTTPError as e:
        return None, f"API HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return None, str(e)

# ── Demo data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_demo_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = datetime.now()
    rows = []
    for i in range(n):
        t      = rng.choice(TRANSACTION_TYPES, p=[0.35, 0.25, 0.20, 0.10, 0.10])
        amount = float(rng.lognormal(7, 2))
        obo    = float(rng.uniform(0, 500_000))
        nbo    = max(0.0, obo - amount) if t in ["PAYMENT","TRANSFER","CASH_OUT","DEBIT"] else obo + amount
        obd    = float(rng.uniform(0, 300_000))
        nbd    = obd + amount if t in ["TRANSFER","CASH_IN"] else obd
        prob   = float(rng.uniform(0.01, 0.15))
        is_fraud = False
        if t in ["TRANSFER","CASH_OUT"] and amount > 200_000:
            prob = float(rng.uniform(0.55, 0.98)); is_fraud = prob > 0.50
        elif obo > 0 and nbo == 0 and amount > 50_000:
            prob = float(rng.uniform(0.45, 0.90)); is_fraud = prob > 0.50
        elif t == "CASH_OUT" and amount > 100_000 and obd == 0:
            prob = float(rng.uniform(0.40, 0.80)); is_fraud = prob > 0.50
        ts = now - timedelta(minutes=int(rng.integers(1, 43200)))
        rows.append({
            "transaction_id":    f"TXN{100000+i:06d}",
            "timestamp":         ts,
            "date":              ts.date(),
            "hour":              ts.hour,
            "day_name":          ts.strftime("%A"),
            "week":              ts.isocalendar()[1],
            "type":              t,
            "amount":            round(amount, 2),
            "oldbalanceOrg":     round(obo, 2),
            "newbalanceOrig":    round(nbo, 2),
            "oldbalanceDest":    round(obd, 2),
            "newbalanceDest":    round(nbd, 2),
            "fraud_probability": round(prob, 4),
            "prediction":        "FRAUD" if is_fraud else "LEGITIMATE",
            "risk_level":        "HIGH" if prob > 0.70 else ("MEDIUM" if prob > 0.40 else "LOW"),
            "sender":            rng.choice(ACCOUNT_NAMES),
            "receiver":          rng.choice(ACCOUNT_NAMES),
            "channel":           rng.choice(CHANNELS),
            "region":            rng.choice(REGIONS),
            "reviewed":          bool(rng.choice([True, False], p=[0.3, 0.7])),
            "confirmed_fraud":   is_fraud and bool(rng.choice([True, False], p=[0.6, 0.4])),
            "balance_drain":     round(max(0.0, obo - nbo), 2),
        })
    return pd.DataFrame(rows)

# ── Chart helpers ─────────────────────────────────────────────────────────────
def apply_chart(fig, height: int = 280, title: str = ""):
    """Apply white Power BI chart theme."""
    fig.update_layout(**CHART_CFG, height=height,
                      title=dict(text=title, font=dict(size=12, color=C["navy"], weight=700), x=0))
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

# ── UI component helpers ──────────────────────────────────────────────────────
def section(title: str, icon: str = ""):
    st.markdown(
        f"<div style='font-size:0.85rem;font-weight:700;color:{C['navy']};"
        f"border-bottom:2px solid {C['navy']};padding-bottom:3px;margin:14px 0 10px;'>"
        f"{icon + ' ' if icon else ''}{title}</div>",
        unsafe_allow_html=True,
    )

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<div style='font-size:1.35rem;font-weight:800;color:{C['navy']};margin-bottom:2px;'>{title}</div>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<div style='font-size:0.78rem;color:{C['grey']};margin-bottom:1rem;'>{subtitle}</div>",
            unsafe_allow_html=True,
        )

def kpi_card(label: str, value: str, sub: str = "", delta=None, color: str = C["navy"]):
    delta_html = ""
    if delta is not None:
        dc = C["red"] if delta > 0 else C["green"]
        arrow = "▲" if delta > 0 else "▼"
        delta_html = f"<div style='font-size:0.7rem;color:{dc};font-weight:600;margin-top:3px;'>{arrow} {abs(delta):.1f}% vs prior period</div>"
    sub_html = f"<div style='font-size:0.68rem;color:{C['grey']};margin-top:1px;'>{sub}</div>" if sub else ""
    return (
        f"<div style='background:#FFFFFF;border-radius:4px;padding:0.9rem 1rem;"
        f"border-left:4px solid {color};box-shadow:0 1px 4px rgba(0,0,0,0.07);'>"
        f"<div style='font-size:0.68rem;color:{C['grey']};text-transform:uppercase;letter-spacing:0.8px;font-weight:600;'>{label}</div>"
        f"<div style='font-size:1.65rem;font-weight:800;color:{color};line-height:1.1;margin-top:4px;'>{value}</div>"
        f"{sub_html}{delta_html}</div>"
    )

def alert_pill(txn_id: str, txn_type: str, amount: float, prob: float,
               ts: str, channel: str = "", region: str = "") -> str:
    col  = C["red"] if prob > 0.70 else C["amber"]
    bw   = int(prob * 100)
    ch   = f"<span style='color:{C['grey']};font-size:0.7rem;'>{channel}</span>" if channel else ""
    rg   = f"<span style='color:{C['grey']};font-size:0.7rem;'> · {region}</span>" if region else ""
    return (
        f"<div style='background:#FFFFFF;border:1px solid {C['lgrey']};border-left:3px solid {col};"
        f"border-radius:4px;padding:0.55rem 0.9rem;margin-bottom:5px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<div><b style='font-size:0.82rem;color:{C['dark']};'>{txn_id}</b>"
        f"<span style='color:{C['grey']};font-size:0.75rem;'> · {txn_type} · KES {amount:,.0f} · {ch}{rg}</span></div>"
        f"<div style='color:{col};font-weight:700;font-size:0.85rem;'>{bw}%</div></div>"
        f"<div style='margin-top:4px;background:{C['lgrey']};border-radius:2px;height:3px;'>"
        f"<div style='width:{bw}%;background:{col};height:3px;border-radius:2px;'></div></div>"
        f"<div style='font-size:0.65rem;color:{C['grey']};margin-top:2px;'>{ts}</div></div>"
    )

def detail_card(fields: dict, border_color: str = C["navy"]) -> str:
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
        f"border-bottom:1px solid {C['lgrey']};'>"
        f"<span style='font-size:0.72rem;color:{C['grey']};'>{k}</span>"
        f"<span style='font-size:0.78rem;font-weight:600;color:{C['dark']};'>{v}</span></div>"
        for k, v in fields.items()
    )
    return (
        f"<div style='background:#FFFFFF;border:1px solid {border_color};"
        f"border-radius:6px;padding:1rem;'>{rows}</div>"
    )

def status_badge(label: str, color: str) -> str:
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color};"
        f"border-radius:12px;padding:2px 10px;font-size:0.72rem;font-weight:700;'>{label}</span>"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo block
        st.markdown(f"""
        <div style='text-align:center;padding:1.4rem 0 1.2rem;
            border-bottom:1px solid rgba(255,255,255,0.12);margin-bottom:1rem;'>
            <div style='font-size:2rem;'>🛡️</div>
            <div style='font-size:0.95rem;font-weight:800;color:#FFFFFF;
                letter-spacing:3px;margin-top:4px;'>FRAUD GUARD</div>
            <div style='font-size:0.58rem;color:#A0B4D0;letter-spacing:2px;margin-top:2px;'>
                MOTOR INSURANCE · DSA 8502</div>
            <div style='font-size:0.58rem;color:#A0B4D0;'>
                Strathmore University · ADM 193665</div>
        </div>""", unsafe_allow_html=True)

        current = st.session_state.get("page", "Home")
        for label, key in PAGES.items():
            is_active = current == key
            st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=lambda k=key: st.session_state.update(page=k),
            )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        # API status badge
        ok, model_name, latency = api_health()
        if ok:
            badge = f"🟢 API Online · {latency}ms"
            badge_color = "#A8D5A8"
        else:
            badge = "🔴 Demo Mode · API offline"
            badge_color = "#E8A0A0"

        st.markdown(
            f"<div style='background:rgba(255,255,255,0.08);border-radius:4px;"
            f"padding:0.5rem 0.7rem;text-align:center;'>"
            f"<div style='font-size:0.68rem;color:{badge_color};font-weight:600;'>{badge}</div>"
            f"{'<div style=\"font-size:0.6rem;color:#A0B4D0;margin-top:2px;\">Model: ' + model_name + '</div>' if ok else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
