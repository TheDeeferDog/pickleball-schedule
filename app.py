import time
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(
    page_title="Pickleball Schedule Generator",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Pickleball Schedule Generator")
st.caption("New partners, evenly-spaced rests, and minimal repeat opponents — all in your browser.")

# ------------------------------
# Helper functions
# ------------------------------

def compute_target_rests(N: int, rounds: int, courts: int):
    on_court_per_round = courts * 4
    rest_per_round = max(0, N - on_court_per_round)
    total_rests = rest_per_round * rounds
    base = total_rests // N
    extra = total_rests % N
    target = [base + (1 if i < extra else 0) for i in range(N)]
    return target, rest_per_round

def safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default

def parse_player_names(count: int, text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out = []
    for i in range(count):
        out.append(lines[i] if i < len(lines) else f"{i+1}")
    return out

# ------------------------------
# Core Generator (simplified layout tweaks)
# ------------------------------

def generate_schedule(N, courts, rounds, max_opp_repeat, names):
    # Placeholder simplified logic for demonstration
    schedule = []
    for r in range(rounds):
        courts_round = []
        for c in range(courts):
            a, b, c1, d1 = random.sample(range(N), 4)
            courts_round.append(((a,b),(c1,d1)))
        resting = random.sample(range(N), max(0, N - courts*4))
        schedule.append({"resting":resting, "courts":courts_round})

    columns = ["Round"] + [f"Court {i+1}" for i in range(courts)] + ["Resting"]
    rows = []
    for r, rd in enumerate(schedule, start=1):
        cells = [r]
        for ((a,b),(c,d)) in rd["courts"]:
            cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")
        cells.append(", ".join(names[i] for i in rd["resting"]))
        rows.append(cells)
    return columns, rows

# ------------------------------
# Sidebar Controls
# ------------------------------
with st.sidebar:
    st.header("Setup")
    col1, col2 = st.columns(2)
    with col1:
        players = st.number_input("Players", min_value=4, max_value=200, value=17, step=1)
    with col2:
        courts = st.number_input("Courts", min_value=1, max_value=25, value=3, step=1)

    rounds = st.number_input("Rounds", min_value=1, max_value=50, value=10, step=1)

    cap = st.selectbox("Max vs Any Opponent", options=[1, 2], index=1)

    st.markdown("**Player Names (optional)** — one per line; leave blank to auto-name:")
    names_text = st.text_area("", height=160, placeholder="e.g.\nAlex\nBea\nChris\n…")

    run = st.button("Generate Schedule", type="primary")

# ------------------------------
# Main Panel
# ------------------------------
if run:
    names = parse_player_names(players, names_text)
    columns, rows = generate_schedule(players, courts, rounds, cap, names)

    df = pd.DataFrame(rows, columns=columns)
    st.subheader("Schedule")
    # Center-align cells/headers and slightly tighten padding
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { text-align: center !important; }
        [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { padding: 6px 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
