import time
import random
from collections import defaultdict, Counter
from typing import List, Tuple

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
# Helpers
# ------------------------------

def compute_target_rests(N: int, rounds: int, courts: int):
    on_court_per_round = courts * 4
    rest_per_round = max(0, N - on_court_per_round)
    total_rests = rest_per_round * rounds
    base = total_rests // N
    extra = total_rests % N
    target = [base + (1 if i < extra else 0) for i in range(N)]
    return target, rest_per_round


def safe_int(s: str, default: int | None) -> int | None:
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
# Core generator with constraint checks & fallback cap escalation
# ------------------------------

def generate_schedule(
    N: int,
    courts: int,
    rounds: int,
    max_opp_repeat: int,
    names: List[str],
    time_budget_sec: float = 4.0,
    seed: int | None = None,
):
    if courts * 4 > N:
        raise ValueError("Courts require more players than available.")
    if N < 4:
        raise ValueError("Need at least 4 players.")

    target_rests, rest_per_round = compute_target_rests(N, rounds, courts)

    # state across attempts
    partner_pairs = [set() for _ in range(N)]
    opponent_counts = [Counter() for _ in range(N)]
    rests_so_far = [0] * N
    last_rest_round = [-99] * N

    schedule: List[dict] = []

    def opp_val(i: int, j: int) -> int:
        return opponent_counts[i][j]

    def inc_opp(i: int, j: int, d: int = 1):
        opponent_counts[i][j] += d
        opponent_counts[j][i] += d

    def choose_resting(rnd: int) -> List[int]:
        # Score-based selection to meet targets and avoid consecutive rests
        scores = []
        for i in range(N):
            if rests_so_far[i] >= target_rests[i]:
                scores.append((-10**9, i)); continue
            if rnd - last_rest_round[i] == 1:
                scores.append((-10**6, i)); continue  # block back-to-back
            need = target_rests[i] - rests_so_far[i]
            since = rnd - last_rest_round[i]
            score = need * 300 + since + random.random()
            scores.append((score, i))
        scores.sort(reverse=True)
        chosen = [i for _, i in scores[:rest_per_round]]
        return chosen

    def pair_players(avail: List[int]) -> List[Tuple[int,int]] | None:
        # Backtracking to form pairs with no repeat partners
        order = sorted(avail, key=lambda x: len(partner_pairs[x]))
        result: List[Tuple[int,int]] = []
        def bt(lst: List[int]) -> bool:
            if not lst: return True
            p = lst[0]
            for k in range(1, len(lst)):
                q = lst[k]
                if q in partner_pairs[p]:
                    continue
                result.append((p,q))
                nxt = lst[1:k] + lst[k+1:]
                if bt(nxt):
                    return True
                result.pop()
            return False
        ok = bt(order)
        return result if ok else None

    def court_cost(court: Tuple[Tuple[int,int],Tuple[int,int]]) -> int | float:
        (a,b), (c,d) = court
        pairs = [(a,c),(a,d),(b,c),(b,d)]
        cost = 0
        for x,y in pairs:
            oc = opp_val(x,y)
            if oc >= 2:  # hard cap: never allow 3+
                return float('inf')
            if oc == 1:
                cost += 1  # try to minimize creating more 2s
        return cost

    def group_pairs_into_courts(pairs: List[Tuple[int,int]]):
        idxs = list(range(len(pairs)))
        best = None; best_cost = float('inf')
        for _ in range(2500):
            random.shuffle(idxs)
            ok = True; cost = 0; courts_round = []
            used = set()
            for i in range(0, len(idxs), 2):
                (a,b) = pairs[idxs[i]]; (c,d) = pairs[idxs[i+1]]
                # redundant safety check: no duplicates
                if len({a,b,c,d}) < 4: ok = False; break
                court = ((a,b),(c,d))
                cst = court_cost(court)
                if not (cst < float('inf')):
                    ok = False; break
                courts_round.append(court); cost += cst
            if ok and cost < best_cost:
                best = courts_round; best_cost = cost
                if best_cost == 0: break
        return best

    def form_courts(avail: List[int]):
        pairs = pair_players(avail)
        if not pairs:
            return None
        return group_pairs_into_courts(pairs)

    def try_build() -> bool:
        # reset per attempt
        for i in range(N):
            partner_pairs[i].clear(); opponent_counts[i].clear()
            rests_so_far[i] = 0; last_rest_round[i] = -99
        schedule.clear()
        base_order = list(range(N)); random.shuffle(base_order)
        for r in range(rounds):
            resting = choose_resting(r)
            rest_set = set(resting)
            # All non-resting must be on court — EXACTLY fill courts without slicing errors
            avail = [i for i in base_order if i not in rest_set]
            if len(avail) != courts * 4:
                return False
            courts_round = form_courts(avail)
            if not courts_round:
                return False
            # apply
            schedule.append({"resting": resting, "courts": courts_round})
            for i in resting:
                rests_so_far[i] += 1; last_rest_round[i] = r
            for (a,b), (c,d) in courts_round:
                partner_pairs[a].add(b); partner_pairs[b].add(a)
                partner_pairs[c].add(d); partner_pairs[d].add(c)
                for x,y in [(a,c),(a,d),(b,c),(b,d)]:
                    inc_opp(x,y,1)
        return True

    if seed is not None:
        random.seed(seed)

    deadline = time.time() + time_budget_sec
    best_snapshot = None; best_score = float('inf')

    while time.time() < deadline:
        if not try_build():
            continue
        # evaluate
        max_opp_by_player = [max(opponent_counts[i].values() or [0]) for i in range(N)]
        violators = sum(1 for m in max_opp_by_player if m > 1)
        total_at_2 = sum(1 for m in max_opp_by_player if m == 2)
        score = violators * 1000 + total_at_2
        if score < best_score:
            best_score = score
            snap_schedule = []
            for rd in schedule:
                snap_schedule.append({
                    "resting": list(rd["resting"]),
                    "courts": [((a,b),(c,d)) for (a,b),(c,d) in rd["courts"]],
                })
            best_snapshot = (snap_schedule, max_opp_by_player)
            if max_opp_repeat == 1 and violators == 0:
                break

    if not best_snapshot:
        return None

    # build table columns/rows (each court one column with both teams)
    schedule_snap, max_opp_by_player = best_snapshot
    columns = ["Round", *[f"Court {k+1}" for k in range(courts)], "Resting"]
    rows = []
    for r, rd in enumerate(schedule_snap, start=1):
        cells = [r]
        for ((a,b),(c,d)) in rd["courts"]:
            cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")
        resting_names = ", ".join(names[i] for i in sorted(rd["resting"]))
        cells.append(resting_names)
        rows.append(cells)

    stats = {
        "oppMax": max_opp_by_player
    }
    return columns, rows, stats, rest_per_round

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

    cap_requested = st.selectbox("Max vs Any Opponent", options=[1, 2], index=1, help="1 is stricter (harder); 2 is recommended.")

    seed_in = st.text_input("Random Seed (optional)")
    seed_val = None if not seed_in.strip() else safe_int(seed_in, None)

    st.markdown("**Player Names (optional)** — one per line; leave blank to auto-name:")
    names_text = st.text_area("", height=140, placeholder="e.g.\nAlex\nBea\nChris\n…")

    run = st.button("Generate Schedule", type="primary")

# ------------------------------
# Main Panel
# ------------------------------
status_box = st.empty()

if run:
    names = parse_player_names(players, names_text)

    # Try requested cap; if it fails, automatically escalate to the minimum cap that works (up to 2)
    tried = []
    result = None
    for cap in [cap_requested, 2]:
        tried.append(cap)
        with st.spinner(f"Building schedule (cap={cap})…"):
            result = generate_schedule(players, courts, rounds, cap, names, seed=seed_val)
        if result is not None:
            effective_cap = cap
            break
    if result is None:
        st.error("Could not construct a schedule with these settings. Try fewer rounds/players or more courts.")
    else:
        columns, rows, stats, rest_per_round = result
        df = pd.DataFrame(rows, columns=columns)
        # Center align cells/headers
        st.markdown(
            """
            <style>
            [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { text-align: center !important; }
            [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { padding: 6px 8px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if effective_cap > cap_requested:
            st.warning(f"Max vs Any Opponent set to {cap_requested} was too strict for a valid schedule. The app used the minimum working cap = {effective_cap} while keeping all other rules.")
        else:
            st.success("Schedule ready!")
        st.subheader("Schedule")
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Download
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="pickleball_schedule.csv", mime="text/csv")
else:
    st.info("Set your event details in the sidebar and click **Generate Schedule**.")
