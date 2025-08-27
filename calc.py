# app.py
# Streamlit: Withdrawal Sell Calculator (Target vs Actual Weights)
# Run: streamlit run app.py

import re
from typing import Tuple
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Withdrawal Sell Calculator", page_icon="💸", layout="centered")

# ---------- Helpers ----------
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
    return e / total, f / total

def round_cents(x: float) -> float:
    return round(float(x) + 1e-9, 2)

def pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return 100.0 * (a - b) / b

# ---------- Core ----------
def calc_sells_target_actual(tav: float, withdrawal: float,
                             w_target_e: float, w_target_f: float,
                             w_actual_e: float, w_actual_f: float) -> dict:
    """
    Compute sells from Equity and FI to move post-withdrawal sleeves toward the target.
    We assume we can only SELL (no buys). If exact target hits with sells-only, use that.
    Otherwise, sell the closest feasible combination (constrained).
    """
    # Basic checks
    if tav <= 0:
        raise ValueError("TAV must be positive.")
    if withdrawal < 0:
        raise ValueError("Withdrawal cannot be negative.")
    if withdrawal > tav:
        raise ValueError("Withdrawal cannot exceed TAV.")
    for w in (w_target_e, w_target_f, w_actual_e, w_actual_f):
        if not (0 <= w <= 1):
            raise ValueError("Weights must be fractions in [0, 1].")
    if abs((w_target_e + w_target_f) - 1) > 1e-6:
        raise ValueError("Target weights must sum to 100%.")
    if abs((w_actual_e + w_actual_f) - 1) > 1e-6:
        raise ValueError("Actual weights must sum to 100%.")

    V0 = float(tav)
    W  = float(withdrawal)
    V1 = V0 - W  # Post-withdrawal TAV

    # Current sleeves (actual)
    E0 = w_actual_e * V0
    F0 = w_actual_f * V0

    # Target sleeves post-withdrawal
    Et = w_target_e * V1
    Ft = w_target_f * V1

    # Unconstrained "exact target" sells
    x_eq = E0 - Et  # sell from equity
    y_fi = F0 - Ft  # sell from FI

    # Feasibility: sells must be >= 0 and <= current sleeves; and sum to W
    sells_sum = round(x_eq + y_fi, 10)
    exact_possible = (x_eq >= -1e-9 and y_fi >= -1e-9 and
                      x_eq <= E0 + 1e-9 and y_fi <= F0 + 1e-9 and
                      abs(sells_sum - W) <= 1e-6)

    if exact_possible:
        sell_e = max(0.0, min(E0, x_eq))
        sell_f = max(0.0, min(F0, y_fi))
        mode = "Exact target achieved with sells-only"
    else:
        # Constrained case: we must sell W dollars total, but cannot buy.
        # Start with the "ideal" sells and clamp negatives to 0 (no buying).
        x = max(0.0, x_eq)
        y = max(0.0, y_fi)

        # Rebalance the pair so x + y == W, respecting sleeve caps.
        # Strategy:
        # 1) If x + y == W after clamping => good, then cap to sleeves if needed.
        # 2) If x + y  < W => allocate the remaining strictly from the sleeve with room.
        # 3) If x + y  > W => reduce proportionally, but not below 0 and not above caps.
        def cap(val, cap_to):  # cap to available sleeve
            return min(max(0.0, val), cap_to)

        # First, cap to sleeves
        x = cap(x, E0)
        y = cap(y, F0)

        s = x + y
        if abs(s - W) <= 1e-6:
            pass  # already good
        elif s < W:
            # Need to add more sells to meet withdrawal
            remaining = W - s
            # Prefer to add from the sleeve that's OVER target (x_eq or y_fi positive), else whichever has room
            room_e = max(0.0, E0 - x)
            room_f = max(0.0, F0 - y)

            # Heuristic: allocate remaining first to the sleeve whose ideal sell was larger
            # (i.e., more above target), then to the other if needed.
            first, second = ("e", "f") if x_eq >= y_fi else ("f", "e")
            if first == "e":
                add = min(remaining, room_e)
                x += add
                remaining -= add
                if remaining > 0:
                    add2 = min(remaining, room_f)
                    y += add2
                    remaining -= add2
            else:
                add = min(remaining, room_f)
                y += add
                remaining -= add
                if remaining > 0:
                    add2 = min(remaining, room_e)
                    x += add2
                    remaining -= add2
            # By construction, remaining should be 0 if W <= TAV.
        else:  # s > W
            # Scale down proportionally to hit total W, but not below zero
            if s > 0:
                scale = W / s
                x *= scale
                y *= scale
            # Re-cap just in case
            x = cap(x, E0)
            y = cap(y, F0)

        sell_e, sell_f = round_cents(x), round_cents(y)
        mode = "Closest feasible to target (sells-only constraint)"

    # Post amounts
    E1 = round_cents(E0 - sell_e)
    F1 = round_cents(F0 - sell_f)

    # Targets (for reference)
    Et = round_cents(Et)
    Ft = round_cents(Ft)

    drift_e = round_cents(pct_diff(E1, Et))
    drift_f = round_cents(pct_diff(F1, Ft))

    return {
        "mode": mode,
        "sell_equity": round_cents(sell_e),
        "sell_fi": round_cents(sell_f),
        "equity_post": E1,
        "fi_post": F1,
        "equity_target_post": Et,
        "fi_target_post": Ft,
        "equity_drift_pct": drift_e,
        "fi_drift_pct": drift_f,
        "tav_post": round_cents(V1),
        "equity_start": round_cents(E0),
        "fi_start": round_cents(F0),
    }

# ---------- UI ----------
st.title("💸 Withdrawal Sell Calculator (Target vs Actual)")
st.caption("Enter TAV, withdrawal, and both target & actual weights (Equity/FI). We’ll compute how much to sell from each sleeve to move you toward target using sells only.")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        tav = st.number_input("Total Account Value (TAV) $", min_value=0.0, step=1000.0, format="%.2f", value=1_000_000.00)
        target_weight_str = st.text_input("Target Weight (Equity/FI) e.g., 70/30", value="70/30")
    with c2:
        withdrawal = st.number_input("Withdrawal Amount $", min_value=0.0, step=1000.0, format="%.2f", value=50_000.00)
        actual_weight_str = st.text_input("Actual Weight (Equity/FI) e.g., 63/37", value="63/37")

    submitted = st.form_submit_button("Calculate")

if submitted:
    try:
        w_te, w_tf = parse_weight(target_weight_str)
        w_ae, w_af = parse_weight(actual_weight_str)

        result = calc_sells_target_actual(
            tav=tav,
            withdrawal=withdrawal,
            w_target_e=w_te,
            w_target_f=w_tf,
            w_actual_e=w_ae,
            w_actual_f=w_af
        )

        st.subheader("Results")

        # Top-line numbers
        m1, m2, m3 = st.columns(3)
        m1.metric("Sell from Equity", f"${result['sell_equity']:,.2f}")
        m2.metric("Sell from Fixed Income", f"${result['sell_fi']:,.2f}")
        m3.metric("Post-Withdrawal TAV", f"${result['tav_post']:,.2f}")

        st.markdown(f"**Method:** {result['mode']}")

        # Detail table
        df = pd.DataFrame(
            {
                "Amount ($)": [
                    result["equity_start"],
                    result["fi_start"],
                    result["equity_post"],
                    result["fi_post"],
                    result["equity_target_post"],
                    result["fi_target_post"],
                ],
                "Drift vs Target (%)": [
                    None,  # N/A for start
                    None,  # N/A for start
                    result["equity_drift_pct"],
                    result["fi_drift_pct"],
                    0.0,  # target rows
                    0.0,
                ],
            },
            index=[
                "Equity Start",
                "FI Start",
                "Equity Post (Actual)",
                "FI Post (Actual)",
                "Equity Target Post",
                "FI Target Post",
            ],
        )

        st.dataframe(
            df.style.format({"Amount ($)": "${:,.2f}", "Drift vs Target (%)": lambda v: "" if v is None else f"{v:+.2f}%"}),
            use_container_width=True
        )

        # Download
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("Download Table (CSV)", data=csv, file_name="withdrawal_sell_results_target_vs_actual.csv", mime="text/csv")

        st.info(
            "If the exact target is reachable with sells only, the calculator sells the exact amounts to hit target post-withdrawal. "
            "If not, it finds the closest feasible split that meets the withdrawal while respecting the no-buy constraint."
        )

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("© Withdrawal Sell Calculator · Uses sells-only mechanics to move toward target after a withdrawal.")
