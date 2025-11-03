import time
import random
from collections import Counter
from typing import List, Tuple, Optional

import pandas as pd
import streamlit as st
from io import BytesIO
# PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

# ------------------------------
# UI CONFIG
# ------------------------------
st.set_page_config(
    page_title="Pickleball Schedule Generator",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Pickleball Schedule Generator")
st.caption("New partners or fixed partners, evenly‑spaced rests (no back‑to‑back), and minimal repeat opponents — all in your browser.")

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


def safe_int(s: str, default: Optional[int]) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return default


def parse_player_names(count: int, text: str) -> List[str]:
    # Default to 1..N labels (no "Player" prefix) unless provided
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out = []
    for i in range(count):
        out.append(lines[i] if i < len(lines) else f"{i+1}")
    return out


def parse_fixed_pairs(N: int, names: List[str], text: str) -> List[Tuple[int, int]]:
    """Parse pairs from textarea. Accept formats like '1 & 2' or 'Alex, Bea'.
    Returns list of (idxA, idxB). If empty, auto‑pair sequentially (0&1, 2&3, ...)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        # Auto-pair sequentially
        pairs = []
        for i in range(0, N - (N % 2), 2):
            pairs.append((i, i + 1))
        return pairs

    # Build lookup by visible name
    name_to_idx = {names[i].strip(): i for i in range(N)}
    used = set()
    pairs: List[Tuple[int, int]] = []
    for ln in lines:
        # split by & or ,
        parts = [p.strip() for p in ln.replace('&', ',').split(',') if p.strip()]
        if len(parts) != 2:
            continue
        a_s, b_s = parts
        # try number
        a = safe_int(a_s, None)
        b = safe_int(b_s, None)
        if a is not None and b is not None:
            a -= 1; b -= 1
        else:
            a = name_to_idx.get(a_s)
            b = name_to_idx.get(b_s)
        if a is None or b is None or a == b or not (0 <= a < N) or not (0 <= b < N):
            continue
        if a in used or b in used:
            continue
        used.add(a); used.add(b)
        pairs.append((a, b))
    return pairs

# ------------------------------
# Core generator with partner modes
# ------------------------------

def generate_schedule(
    N: int,
    courts: int,
    rounds: int,
    max_opp_repeat: int,
    names: List[str],
    partner_mode: str,  # 'rotate' or 'fixed'
    fixed_pairs: Optional[List[Tuple[int, int]]] = None,
    time_budget_sec: float = 4.0,
    seed: Optional[int] = None,
):
    if courts * 4 > N:
        raise ValueError("Courts require more players than available.")
    if N < 4:
        raise ValueError("Need at least 4 players.")

    target_rests, rest_per_round = compute_target_rests(N, rounds, courts)

    # If fixed partners, we must rest in PAIRS. That requires an even number of rest slots per round.
    if partner_mode == 'fixed' and (rest_per_round % 2 != 0):
        raise ValueError(
            "Fixed partner mode requires an even number of resting players each round. "
            "Adjust players/courts so N - 4*courts is even, or use Rotate partners."
        )

    # State for attempts
    partner_pairs = [set() for _ in range(N)]  # used only in rotate mode
    opponent_counts = [Counter() for _ in range(N)]
    rests_so_far = [0] * N
    last_rest_round = [-99] * N

    schedule = []  # list of {resting:[idx], courts:[((a,b),(c,d)), ...]}

    def opp_val(i: int, j: int) -> int:
        return opponent_counts[i][j]

    def inc_opp(i: int, j: int, d: int = 1):
        opponent_counts[i][j] += d
        opponent_counts[j][i] += d

    # ------------- Rest selection -------------
    def choose_resting_rotate(rnd: int) -> List[int]:
        # per-player scoring
        scores = []
        for i in range(N):
            if rests_so_far[i] >= target_rests[i]:
                scores.append((-10**9, i)); continue
            if rnd - last_rest_round[i] == 1:
                scores.append((-10**6, i)); continue
            need = target_rests[i] - rests_so_far[i]
            since = rnd - last_rest_round[i]
            scores.append((need * 300 + since + random.random(), i))
        scores.sort(reverse=True)
        chosen = [i for _, i in scores[:rest_per_round]]
        return chosen

    def choose_resting_fixed(rnd: int, pair_list: List[Tuple[int, int]]) -> List[int]:
        # score by PAIRS (both rest together)
        pairs_scores = []
        pairs_needed = rest_per_round // 2
        for a, b in pair_list:
            # If either has maxed rests, deprioritize; block back-to-back if either rested last round
            if rests_so_far[a] >= target_rests[a] or rests_so_far[b] >= target_rests[b]:
                score = -10**9
            elif rnd - last_rest_round[a] == 1 or rnd - last_rest_round[b] == 1:
                score = -10**6
            else:
                need = (target_rests[a] - rests_so_far[a]) + (target_rests[b] - rests_so_far[b])
                since = min(rnd - last_rest_round[a], rnd - last_rest_round[b])
                score = need * 300 + since + random.random()
            pairs_scores.append((score, (a, b)))
        pairs_scores.sort(reverse=True)
        chosen_pairs = [p for _, p in pairs_scores[:pairs_needed]]
        chosen = [x for pair in chosen_pairs for x in pair]
        return chosen

    # ------------- Pair & court formation -------------
    def pair_players_rotate(avail: List[int]) -> Optional[List[Tuple[int, int]]]:
        # Backtracking to ensure no repeat partners
        order = sorted(avail, key=lambda x: len(partner_pairs[x]))
        result: List[Tuple[int, int]] = []
        def bt(lst: List[int]) -> bool:
            if not lst: return True
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

    def pairs_for_fixed(avail: List[int], pair_list: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        # Simply pick the subset of predeclared pairs that are available (entire set should be available if resting chosen by pairs)
        avail_set = set(avail)
        pairs = [(a, b) for (a, b) in pair_list if a in avail_set and b in avail_set]
        # Sanity: length should match
        return pairs if len(pairs) * 2 == len(avail) else None

    def court_cost(court: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        (a, b), (c, d) = court
        cost = 0
        for x, y in [(a, c), (a, d), (b, c), (b, d)]:
            oc = opp_val(x, y)
            if oc >= 2:  # hard stop at 3+
                return float('inf')
            if oc == 1:
                cost += 1
        return cost

    def group_pairs_into_courts(pairs: List[Tuple[int, int]]) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        idxs = list(range(len(pairs)))
        best = None; best_cost = float('inf')
        for _ in range(3000):
            random.shuffle(idxs)
            ok = True; cost = 0; courts_round = []
            for i in range(0, len(idxs), 2):
                (a, b) = pairs[idxs[i]]; (c, d) = pairs[idxs[i + 1]]
                if len({a, b, c, d}) < 4: ok = False; break
                court = ((a, b), (c, d))
                cst = court_cost(court)
                if not (cst < float('inf')):
                    ok = False; break
                courts_round.append(court); cost += cst
            if ok and cost < best_cost:
                best = courts_round; best_cost = cost
                if best_cost == 0: break
        return best

    def form_courts(avail: List[int], pair_list: Optional[List[Tuple[int, int]]] = None):
        if partner_mode == 'fixed':
            pairs = pairs_for_fixed(avail, pair_list or [])
        else:
            pairs = pair_players_rotate(avail)
        if not pairs:
            return None
        return group_pairs_into_courts(pairs)

    # ------------- Attempt loop -------------
    if seed is not None:
        random.seed(seed)

    # If fixed partners, define list
    declared_pairs: List[Tuple[int, int]] = []
    if partner_mode == 'fixed':
        if fixed_pairs:
            declared_pairs = fixed_pairs
        else:
            declared_pairs = [(i, i + 1) for i in range(0, N - (N % 2), 2)]
        # Validate no overlaps
        seen = set()
        for a, b in declared_pairs:
            if a in seen or b in seen or a == b:
                raise ValueError("Fixed pairs input has overlapping/invalid players.")
            seen.add(a); seen.add(b)
        if len(seen) < N and (N - len(seen)) % 2 != 0:
            raise ValueError("Unpaired players remain; please complete pairs or reduce player count.")

    deadline = time.time() + time_budget_sec
    best_snapshot = None; best_score = float('inf')

    while time.time() < deadline:
        # reset per attempt
        for i in range(N):
            partner_pairs[i].clear()
            opponent_counts[i].clear()
            rests_so_far[i] = 0
            last_rest_round[i] = -99
        schedule.clear()

        base_order = list(range(N))
        random.shuffle(base_order)

        feasible = True
        for r in range(rounds):
            if partner_mode == 'fixed':
                resting = choose_resting_fixed(r, declared_pairs)
            else:
                resting = choose_resting_rotate(r)
            rest_set = set(resting)
            avail = [i for i in base_order if i not in rest_set]
            if len(avail) != courts * 4:
                feasible = False; break
            courts_round = form_courts(avail, declared_pairs if partner_mode == 'fixed' else None)
            if not courts_round:
                feasible = False; break
            # apply
            schedule.append({"resting": resting, "courts": courts_round})
            for i in resting:
                rests_so_far[i] += 1; last_rest_round[i] = r
            for (a, b), (c, d) in courts_round:
                if partner_mode == 'rotate':
                    partner_pairs[a].add(b); partner_pairs[b].add(a)
                    partner_pairs[c].add(d); partner_pairs[d].add(c)
                for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                    opponent_counts[x][y] += 1
                    opponent_counts[y][x] += 1
        if not feasible:
            continue

        # evaluate
        max_opp_by_player = [max(opponent_counts[i].values() or [0]) for i in range(N)]
        violators = sum(1 for m in max_opp_by_player if m > 1)
        total_at_2 = sum(1 for m in max_opp_by_player if m == 2)
        score = violators * 1000 + total_at_2
        if score < best_score:
            best_score = score
            best_snapshot = ([{ "resting": list(rd["resting"]), "courts": [((a,b),(c,d)) for (a,b),(c,d) in rd["courts"]] } for rd in schedule], max_opp_by_player)
            if max_opp_repeat == 1 and violators == 0:
                break

    if not best_snapshot:
        return None

    snap, max_opp = best_snapshot

    # Build table (each court one column)
    columns = ["Round", *[f"Court {k+1}" for k in range(courts)], "Resting"]
    rows = []
    for r, rd in enumerate(snap, start=1):
        cells = [r]
        for ((a, b), (c, d)) in rd["courts"]:
            cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")
        cells.append(", ".join(names[i] for i in sorted(rd["resting"])))
        rows.append(cells)

    stats = {"oppMax": max_opp}
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

    partner_mode = st.radio(
        "Partner mode",
        options=["Rotate partners (all different)", "Stick with same partner"],
        index=0,
    )
    use_fixed = partner_mode.startswith("Stick")

    cap_requested = st.selectbox("Max vs Any Opponent", options=[1, 2], index=1, help="1 is stricter (harder); 2 is recommended.")

    seed_in = st.text_input("Random Seed (optional)")
    seed_val = None if not seed_in.strip() else safe_int(seed_in, None)

    st.markdown("**Player Names (optional)** — one per line; leave blank to auto-name 1..N.")
    names_text = st.text_area("", height=120, placeholder="e.g.\nAlex\nBea\nChris\n…")

    fixed_pairs_text = ""
    if use_fixed:
        st.markdown("**Fixed pairs (optional)** — one pair per line, e.g. `1 & 2` or `Alex, Bea`. Leave blank to auto‑pair sequentially (1&2, 3&4, …).")
        fixed_pairs_text = st.text_area("Pairs", height=140)

    run = st.button("Generate Schedule", type="primary")

# ------------------------------
# PDF helper
# ------------------------------

def build_print_pdf(df: pd.DataFrame, title: str = "Pickleball Schedule", big: bool = True) -> bytes:
    """Render the schedule to a Letter landscape PDF with large font.
    If there are many court columns (e.g., 6 courts), automatically split
    across multiple pages so each page shows at most 3 courts + Resting
    with large, legible text.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=24,
        rightMargin=24,
        topMargin=28,
        bottomMargin=28,
    )
    story = []

    styles = getSampleStyleSheet()
    h_style = styles["Heading1"].clone("H1")
    h_style.fontSize = 22 if big else 18
    h_style.leading = h_style.fontSize + 4

    cell_style = styles["BodyText"].clone("Cell")
    cell_style.fontName = 'Helvetica'
    cell_style.fontSize = 16 if big else 13
    cell_style.leading = cell_style.fontSize + 4
    cell_style.alignment = 1  # TA_CENTER

    header_size = 17 if big else 14

    # Identify columns
    cols = list(df.columns)
    round_col = cols[0]
    rest_col = cols[-1]
    court_cols = [c for c in cols[1:-1] if c.lower().startswith("court ")]

    # Chunk court columns into groups of up to 3 per page for legibility
    chunk_size = 3
    chunks = [court_cols[i:i+chunk_size] for i in range(0, len(court_cols), chunk_size)] or [[]]

    for page_idx, group in enumerate(chunks):
        page_cols = [round_col] + group + [rest_col]
        # Title per page indicating which courts are shown
        if group:
            court_range = f" (Courts {group[0].split()[-1]}–{group[-1].split()[-1]})"
        else:
            court_range = ""
        story.append(Paragraph(title + court_range, h_style))
        story.append(Spacer(1, 8))

        # Build table data as Paragraphs for better wrapping/centering
        data = [page_cols]
        for _, row in df.iterrows():
            data.append([Paragraph(str(row[c]), cell_style) for c in page_cols])

        # Compute column widths: give Resting more width
        page_width, _ = landscape(letter)
        avail_width = page_width - 48
        col_count = len(page_cols)
        # Base width with extra for Resting
        base = avail_width / col_count
        col_widths = [base for _ in range(col_count)]
        # Expand Resting and slightly shrink others
        rest_idx = page_cols.index(rest_col)
        col_widths[rest_idx] = base * 1.8
        reduce_each = (col_widths[rest_idx] - base) / (col_count - 1)
        for i in range(col_count):
            if i == rest_idx:
                continue
            col_widths[i] = max(base - reduce_each, 80)

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), header_size),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))

        story.append(tbl)
        if page_idx < len(chunks) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()

# ------------------------------
# Main Panel
# ------------------------------
status_box = st.empty()

if run:
    names = parse_player_names(players, names_text)
    fixed_pairs = None
    if use_fixed:
        fixed_pairs = parse_fixed_pairs(players, names, fixed_pairs_text)

    # Try requested cap; if impossible, escalate to 2 and warn
    result = None
    for cap in [cap_requested, 2]:
        try:
            with st.spinner(f"Building schedule (cap={cap})…"):
                result = generate_schedule(
                    N=players,
                    courts=courts,
                    rounds=rounds,
                    max_opp_repeat=cap,
                    names=names,
                    partner_mode='fixed' if use_fixed else 'rotate',
                    fixed_pairs=fixed_pairs,
                    seed=seed_val,
                )
        except ValueError as e:
            st.error(str(e))
            result = None
            break
        if result is not None:
            effective_cap = cap
            break

    if result is None:
        st.error("Could not construct a schedule with these settings. Try adjusting players/courts/rounds.")
    else:
        columns, rows, stats, rest_per_round = result
        df = pd.DataFrame(rows, columns=columns)
        # Center align cells and headers
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
            st.warning(f"Max vs Any Opponent = {cap_requested} was too strict; using minimum working cap = {effective_cap} while keeping all other rules.")
        else:
            st.success("Schedule ready!")
        st.subheader("Schedule")
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        col_a, col_b = st.columns([1,1])
        with col_a:
            st.download_button("Download CSV", csv_bytes, file_name="pickleball_schedule.csv", mime="text/csv")
        with col_b:
            big = st.checkbox("Large print (recommended)", value=True)
            pdf_bytes = build_print_pdf(df, title="Pickleball Schedule", big=big)  # auto-splits wide tables across pages for large print
                    st.subheader("Schedule")         st.dataframe(df, use_container_width=True, hide_index=True)          # Download (CSV + Print-friendly PDF)         csv_bytes = df.to_csv(index=False).encode("utf-8")         col_a, col_b = st.columns([1, 1])          with col_a:             st.download_button(                 "Download CSV",                 data=csv_bytes,                 file_name="pickleball_schedule.csv",                 mime="text/csv",             )          with col_b:             big = st.checkbox("Large print (recommended)", value=True)             pdf_bytes = build_print_pdf(df, title="Pickleball Schedule", big=big)             st.download_button(                 "Get Print Version (PDF)",                 data=pdf_bytes,                 file_name="pickleball_schedule_print.pdf",                 mime="application/pdf",             )
else:
    st.info("Set your event details in the sidebar and click **Generate Schedule**. Use ‘Stick with same partner’ to keep fixed teams (requires an even number of resting players each round)."}]}
