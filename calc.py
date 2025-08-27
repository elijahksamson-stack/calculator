# app.py
# Streamlit: Withdrawal Sell Calculator
# Run: streamlit run app.py

import re
from typing import Tuple
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Withdrawal Sell Calculator", page_icon="💸", layout="centered")

# ---------- Core Logic (from your script) ----------
def parse_weight(weight_str: str) -> Tuple[float, float]:
    """
    Parse "70/30", "70-30", or "70,30" as Equity/Fixed Income weights.
    Returns (w_equity, w_fi) as fractions that sum to 1.0.
    """
    if not isinstance(weight_str, str):
        raise ValueError("Weight must be a string like '70/30'.")
    parts = re.split(r"[\/,\-\s]+", weight_str.strip())
    if len(parts) != 2:
        raise ValueError("Weight should look like '70/30' (Equity/FI).")
    try:
        e, f = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError("Weights must be numbers, e.g., '70/30'.")
    if e < 0 or f < 0 or (e + f) == 0:
        raise ValueError("Weights must be non-negative and not both zero.")
    total = e + f
    w_e = e / total
    w_f = f / total
    return w_e, w_f

def round_cents(x: float) -> float:
    return round(x + 1e-9, 2)

def calc_sells(tav: float, withdrawal: float, w_e: float, w_f: float, policy: str = "proportional") -> dict:
    """
    Compute how much to sell from Equity and FI for the given policy.

    Assumption for 'proportional': starting sleeves are at target (w_e*tav, w_f*tav),
    and we want to remain at target after the withdrawal.
    """
    if tav <= 0:
        raise ValueError("TAV must be positive.")
    if withdrawal < 0:
        raise ValueError("Withdrawal cannot be negative.")
    if withdrawal > tav:
        raise ValueError("Withdrawal cannot exceed TAV.")
    if not (0 <= w_e <= 1 and 0 <= w_f <= 1) or abs((w_e + w_f) - 1) > 1e-6:
        raise ValueError("Weights must be fractions that sum to 1.")

    V = float(tav)
    W = float(withdrawal)
    V_post = V - W

    # Starting sleeves at target (operational assumption)
    eq_start = w_e * V
    fi_start = w_f * V

    policy = policy.lower().strip()
    if policy not in {"proportional", "equity", "fi"}:
        raise ValueError("Policy must be 'proportional', 'equity', or 'fi'.")

    if policy == "proportional":
        sell_eq = w_e * W
        sell_fi = w_f * W
    elif policy == "equity":
        sell_eq = W
        sell_fi = 0.0
    else:  # policy == "fi"
        sell_eq = 0.0
        sell_fi = W

    eq_post = max(eq_start - sell_eq, 0.0)
    fi_post = max(fi_start - sell_fi, 0.0)

    # Targets for comparison
    eq_target_post = w_e * V_post
    fi_target_post = w_f * V_post

    # % drift vs target (useful if not proportional)
    def pct_diff(a, b):
        if b == 0:
            return 0.0
        return 100.0 * (a - b) / b

    return {
        "policy": policy,
        "sell_equity": round_cents(sell_eq),
        "sell_fi": round_cents(sell_fi),
        "equity_post": round_cents(eq_post),
        "fi_post": round_cents(fi_post),
        "equity_target_post": round_cents(eq_target_post),
        "fi_target_post": round_cents(fi_target_post),
        "equity_drift_pct": round(round_cents(pct_diff(eq_post, eq_target_post)), 2),
        "fi_drift_pct": round(round_cents(pct_diff(fi_post, fi_target_post)), 2),
        "tav_post": round_cents(V_post),
    }

# ---------- UI ----------
st.title("💸 Withdrawal Sell Calculator")
st.caption("Enter TAV, withdrawal, and tactical weights (Equity/FI). Choose a sell policy and get exact sell amounts.")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        tav = st.number_input("Total Account Value (TAV) $", min_value=0.0, step=1000.0, format="%.2f", value=1_000_000.00)
        weight_str = st.text_input("Tactical Weight (Equity/FI) e.g., 70/30", value="70/30")
    with col2:
        withdrawal = st.number_input("Withdrawal Amount $", min_value=0.0, step=1000.0, format="%.2f", value=50_000.00)
        policy = st.selectbox("Sell Policy", options=["proportional", "equity", "fi"], index=0,
                              help="• proportional: sell by targets\n• equity: sell all from equity\n• fi: sell all from fixed income")

    submitted = st.form_submit_button("Calculate")

if submitted:
    try:
        w_e, w_f = parse_weight(weight_str)
        result = calc_sells(tav, withdrawal, w_e, w_f, policy)

        st.subheader("Results")

        # Top-line numbers
        m1, m2, m3 = st.columns(3)
        m1.metric("Sell from Equity", f"${result['sell_equity']:,.2f}")
        m2.metric("Sell from Fixed Income", f"${result['sell_fi']:,.2f}")
        m3.metric("Post-Withdrawal TAV", f"${result['tav_post']:,.2f}")

        # Detail table
        df = pd.DataFrame(
            {
                "Amount ($)": [
                    result["equity_post"],
                    result["fi_post"],
                    result["equity_target_post"],
                    result["fi_target_post"],
                ],
                "Drift vs Target (%)": [
                    result["equity_drift_pct"],
                    result["fi_drift_pct"],
                    0.0,
                    0.0,
                ],
            },
            index=["Equity Post", "FI Post", "Equity Target Post", "FI Target Post"],
        )
        st.dataframe(df.style.format({"Amount ($)": "${:,.2f}", "Drift vs Target (%)": "{:+.2f}%"}),
                     use_container_width=True)

        # Download
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("Download Table (CSV)", data=csv, file_name="withdrawal_sell_results.csv", mime="text/csv")

        # Notes
        st.info(
            "Proportional policy keeps sleeves exactly at target after withdrawal. "
            "Equity/FI-only policies may leave a drift relative to target."
        )

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("© Streamlit Withdrawal Sell Calculator · Assumes sleeves are at target before withdrawal.")
