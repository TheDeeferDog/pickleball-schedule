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
        out.append(lines[i] if i < len(lines) else f"Player {i+1}")
    return out


# ------------------------------
# Core Generator
# ------------------------------

def generate_schedule(
    N: int,
    courts: int,
    rounds: int,
    max_opp_repeat: int,
    names: List[str],
    time_budget_sec: float = 3.5,
    seed: int | None = None,
):
    """Return (columns, rows, stats_dict). May raise ValueError on impossible inputs."""

    if courts * 4 > N:
        raise ValueError("Courts require more players than available.")
    if N < 4:
        raise ValueError("Need at least 4 players.")

    target_rests, rest_per_round = compute_target_rests(N, rounds, courts)

    # Global state containers (mutable in attempts)
    partner_pairs = [set() for _ in range(N)]  # i -> set(partners)
    opponent_counts = [Counter() for _ in range(N)]  # i -> Counter({j: times})
    rests_so_far = [0] * N
    last_rest_round = [-99] * N

    schedule: List[Dict] = []

    def opp_val(i: int, j: int) -> int:
        return opponent_counts[i][j]

    def inc_opp(i: int, j: int, delta: int = 1):
        opponent_counts[i][j] += delta
        opponent_counts[j][i] += delta

    def choose_resting(rnd: int) -> List[int]:
        scores = []
        for i in range(N):
            if rests_so_far[i] >= target_rests[i]:
                scores.append((-10**9, i))
                continue
            if rnd - last_rest_round[i] == 1:
                scores.append((-10**6, i))  # hard block back-to-back rests
                continue
            need = target_rests[i] - rests_so_far[i]
            since = rnd - last_rest_round[i]
            # jitter helps diversify choices between attempts
            score = need * 300 + since + random.random()
            scores.append((score, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:rest_per_round]]

    def pair_players(avail: List[int]) -> List[Tuple[int, int]] | None:
        # Backtracking to form pairs with no repeat partners
        order = sorted(avail, key=lambda x: len(partner_pairs[x]))
        result: List[Tuple[int, int]] = []

        def bt(lst: List[int]) -> bool:
            if not lst:
                return True
            p = lst[0]
            for k in range(1, len(lst)):
                q = lst[k]
                if q in partner_pairs[p]:
                    continue
                result.append((p, q))
                nxt = lst[1:k] + lst[k+1:]
                if bt(nxt):
                    return True
                result.pop()
            return False

        ok = bt(order)
        return result if ok else None

    def court_soft_cost(court: Tuple[Tuple[int, int], Tuple[int, int]]) -> int | float:
        (a, b), (c, d) = court
        pairs = [(a, c), (a, d), (b, c), (b, d)]
        cost = 0
        for x, y in pairs:
            oc = opp_val(x, y)
            if oc >= 2:
                return float("inf")  # never allow 3+
            if oc == 1:
                # We minimize creating more 2s regardless, but if strict cap=1,
                # producing a 2 is especially undesirable.
                cost += 1
        return cost

    def group_pairs_into_courts(pairs: List[Tuple[int, int]]):
        idxs = list(range(len(pairs)))
        best = None
        best_cost = float("inf")
        for _ in range(2000):
            random.shuffle(idxs)
            courts_round = []
            ok = True
            total_cost = 0
            for i in range(0, len(idxs), 2):
                (a, b) = pairs[idxs[i]]
                (c, d) = pairs[idxs[i + 1]]
                court = ((a, b), (c, d))
                cst = court_soft_cost(court)
                if not (cst < float("inf")):
                    ok = False
                    break
                courts_round.append(court)
                total_cost += cst
            if ok:
                # Validate no player duplication & no repeat partners (already avoided), & soft cost best
                if total_cost < best_cost:
                    best_cost = total_cost
                    best = courts_round
                    if best_cost == 0:
                        break
        return best

    def form_courts(avail: List[int]):
        pairs = pair_players(avail)
        if not pairs:
            return None
        return group_pairs_into_courts(pairs)

    def try_build() -> bool:
        # reset per attempt
        for i in range(N):
            partner_pairs[i].clear()
            opponent_counts[i].clear()
            rests_so_far[i] = 0
            last_rest_round[i] = -99
        schedule.clear()

        base_order = list(range(N))
        random.shuffle(base_order)

        for r in range(rounds):
            resting = choose_resting(r)
            rest_set = set(resting)
            avail = [i for i in base_order if i not in rest_set][: courts * 4]
            if len(avail) != courts * 4:
                return False
            courts_round = form_courts(avail)
            if not courts_round:
                return False
            schedule.append({"resting": resting, "courts": courts_round})
            for i in resting:
                rests_so_far[i] += 1
                last_rest_round[i] = r
            for (a, b), (c, d) in courts_round:
                partner_pairs[a].add(b)
                partner_pairs[b].add(a)
                partner_pairs[c].add(d)
                partner_pairs[d].add(c)
                for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                    inc_opp(x, y, 1)
        return True

    if seed is not None:
        random.seed(seed)

    deadline = time.time() + time_budget_sec
    best_snapshot = None
    best_score = float("inf")

    while time.time() < deadline:
        if not try_build():
            continue
        # evaluate
        max_opp_by_player = [0] * N
        for i in range(N):
            if opponent_counts[i]:
                max_opp_by_player[i] = max(opponent_counts[i].values())
            else:
                max_opp_by_player[i] = 0
        violators = sum(1 for m in max_opp_by_player if m > 1)
        total_at_2 = sum(1 for m in max_opp_by_player if m == 2)
        score = violators * 1000 + total_at_2
        # If strict cap requested and we hit zero violators, we're done
        if score < best_score:
            best_score = score
            # Deep copy the useful bits
            snap_schedule = []
            for rd in schedule:
                snap_schedule.append({
                    "resting": list(rd["resting"]),
                    "courts": [((a, b), (c, d)) for (a, b), (c, d) in rd["courts"]],
                })
            snap_opponents = [Counter(opp) for opp in opponent_counts]
            best_snapshot = (snap_schedule, snap_opponents)
            if max_opp_repeat == 1 and violators == 0:
                break

    if not best_snapshot:
        raise ValueError("Could not construct a schedule with the given settings. Try reducing players or rounds, or set Max vs Opponent to 2.")

    # Render using names
    schedule, opp_counts = best_snapshot

    columns = [
        "Round",
        *[f"Court {k+1} (Team A)" for k in range(courts)],
        *[f"Court {k+1} (Team B)" for k in range(courts)],
        f"Resting ({rest_per_round})",
    ]

    def idx_to_name(i: int) -> str:
        return names[i] if 0 <= i < len(names) else f"Player {i+1}"

    rows = []
    for r, rd in enumerate(schedule, start=1):
        teamsA, teamsB = [], []
        for ((a, b), (c, d)) in rd["courts"]:
            teamsA.append(f"{idx_to_name(a)} & {idx_to_name(b)}")
            teamsB.append(f"{idx_to_name(c)} & {idx_to_name(d)}")
        resting_names = ", ".join(idx_to_name(i) for i in sorted(rd["resting"]))
        rows.append([r, *teamsA, *teamsB, resting_names])

    # Stats recompute from snapshot
    games = Counter({i: 0 for i in range(N)})
    rests = Counter({i: 0 for i in range(N)})
    partners_hist = [set() for _ in range(N)]
    opp_max = [0] * N

    for rd in schedule:
        for i in rd["resting"]:
            rests[i] += 1
        for (a, b), (c, d) in rd["courts"]:
            games[a] += 1; games[b] += 1; games[c] += 1; games[d] += 1
            partners_hist[a].add(b); partners_hist[b].add(a)
            partners_hist[c].add(d); partners_hist[d].add(c)
    for i in range(N):
        opp_max[i] = max(opp_counts[i].values() or [0])

    stats = {
        "games": [games[i] for i in range(N)],
        "rests": [rests[i] for i in range(N)],
        "partners": [len(partners_hist[i]) for i in range(N)],
        "oppMax": opp_max,
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

    cap = st.selectbox("Max vs Any Opponent", options=[1, 2], index=1, help="1 is stricter (harder to satisfy); 2 is recommended.")

    seed_in = st.text_input("Random Seed (optional)")
    seed_val = None if not seed_in.strip() else safe_int(seed_in, None)

    st.markdown("**Player Names (optional)** — one per line; leave blank to auto-name:")
    names_text = st.text_area("", height=160, placeholder="e.g.\nAlex\nBea\nChris\n…")

    st.markdown("— The generator enforces **new partners**, **no back-to-back rests** with even spread, and **minimizes repeats** (hard-caps 3+ to impossible; your selected cap guides the optimizer).")

    run = st.button("Generate Schedule", type="primary")


# ------------------------------
# Main Panel
# ------------------------------
placeholder_status = st.empty()
placeholder_stats = st.empty()
placeholder_table = st.empty()
placeholder_download = st.empty()

if run:
    try:
        with st.spinner("Building a fair schedule…"):
            names = parse_player_names(players, names_text)
            columns, rows, stats, rest_per_round = generate_schedule(
                N=players,
                courts=courts,
                rounds=rounds,
                max_opp_repeat=cap,
                names=names,
                time_budget_sec=4.0,
                seed=seed_val,
            )

        # Status / KPIs
        violators = sum(1 for m in stats["oppMax"] if m > 1)
        worst = max(stats["oppMax"]) if stats["oppMax"] else 0
        avg_games = sum(stats["games"]) / players if players else 0

        status_text = "Schedule ready!"
        if cap == 1 and violators > 0:
            status_text = (
                f"Generated with minimal repeats: **{violators} player(s)** have max opponent = **{worst}**. "
                "Consider switching cap to 2 for easier scheduling."
            )
        placeholder_status.info(status_text)

        # Table
        df = pd.DataFrame(rows, columns=columns)
        with placeholder_table.container():
            st.subheader("Schedule")
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        placeholder_download.download_button(
            "Download CSV",
            data=csv,
            file_name="pickleball_schedule.csv",
            mime="text/csv",
        )

        # Stats table
        stats_df = pd.DataFrame({
            "Player": [names[i] for i in range(players)],
            "Games": stats["games"],
            "Rests": stats["rests"],
            "Unique Partners": stats["partners"],
            "Max vs Any Opponent": stats["oppMax"],
        })
        with placeholder_stats.container():
            st.subheader("Player Stats")
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    except Exception as e:
        placeholder_status.error(str(e))
        placeholder_table.empty()
        placeholder_download.empty()
        placeholder_stats.empty()

else:
    st.info("Set your event details in the sidebar and click **Generate Schedule**.")
