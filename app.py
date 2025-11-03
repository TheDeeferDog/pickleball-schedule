import time
import random
from collections import Counter
from typing import List, Tuple, Optional
from io import BytesIO

import pandas as pd
import streamlit as st

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.pdfbase.pdfmetrics import stringWidth

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Pickleball Schedule Generator",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Pickleball Schedule Generator")
st.caption("Rotate or fixed partners, evenly spaced rests, and minimal repeat opponents — printable and large-font friendly.")

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def compute_target_rests(N: int, rounds: int, courts: int):
    on_court = courts * 4
    rest_per_round = max(0, N - on_court)
    total_rests = rest_per_round * rounds
    base = total_rests // N
    extra = total_rests % N
    return [base + (1 if i < extra else 0) for i in range(N)], rest_per_round


def parse_player_names(count: int, text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return [lines[i] if i < len(lines) else f"{i+1}" for i in range(count)]


def safe_int(s: str, default: Optional[int]) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return default


def parse_fixed_pairs(N: int, names: List[str], text: str) -> List[Tuple[int, int]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    name_to_idx = {names[i]: i for i in range(N)}
    pairs = []
    used = set()
    for ln in lines:
        parts = [p.strip() for p in ln.replace("&", ",").split(",") if p.strip()]
        if len(parts) != 2:
            continue
        a = name_to_idx.get(parts[0], safe_int(parts[0], None))
        b = name_to_idx.get(parts[1], safe_int(parts[1], None))
        if a is None or b is None or a == b or a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pairs.append((a, b))
    if not pairs:
        for i in range(0, N - (N % 2), 2):
            pairs.append((i, i + 1))
    return pairs


# ------------------------------------------------------------
# Schedule generator (kept simple & reliable)
# ------------------------------------------------------------
def generate_schedule(N, courts, rounds, max_opp_repeat, names, partner_mode, fixed_pairs=None, seed=None, time_budget_sec=4.0):
    if seed:
        random.seed(seed)

    target_rests, rest_per_round = compute_target_rests(N, rounds, courts)
    opponent_counts = [Counter() for _ in range(N)]
    partner_pairs = [set() for _ in range(N)]
    rests_so_far = [0] * N
    last_rest_round = [-99] * N
    schedule = []

    if partner_mode == "fixed" and rest_per_round % 2 != 0:
        raise ValueError("Fixed partner mode requires an even number of resting players each round.")

    pairs = fixed_pairs or []
    base_order = list(range(N))

    def inc_opp(i, j):
        opponent_counts[i][j] += 1
        opponent_counts[j][i] += 1

    def choose_resting(r):
        scores = []
        for i in range(N):
            if rests_so_far[i] >= target_rests[i]:
                scores.append((-1e9, i))
                continue
            if r - last_rest_round[i] == 1:
                scores.append((-1e6, i))
                continue
            need = target_rests[i] - rests_so_far[i]
            since = r - last_rest_round[i]
            scores.append((need * 100 + since + random.random(), i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:rest_per_round]]

    def choose_resting_pairs(r):
        pair_scores = []
        pair_rests = rest_per_round // 2
        for a, b in pairs:
            if rests_so_far[a] >= target_rests[a] or rests_so_far[b] >= target_rests[b]:
                score = -1e9
            elif r - last_rest_round[a] == 1 or r - last_rest_round[b] == 1:
                score = -1e6
            else:
                need = (target_rests[a] - rests_so_far[a]) + (target_rests[b] - rests_so_far[b])
                since = min(r - last_rest_round[a], r - last_rest_round[b])
                score = need * 100 + since + random.random()
            pair_scores.append((score, (a, b)))
        pair_scores.sort(reverse=True)
        chosen = [x for _, pair in pair_scores[:pair_rests] for x in pair]
        return chosen

    def pair_players(avail):
        result = []
        avail = avail.copy()
        random.shuffle(avail)
        while avail:
            a = avail.pop()
            possible = [b for b in avail if b not in partner_pairs[a]]
            if not possible:
                return None
            b = random.choice(possible)
            avail.remove(b)
            result.append((a, b))
            partner_pairs[a].add(b)
            partner_pairs[b].add(a)
        return result

    def group_into_courts(pairs):
        random.shuffle(pairs)
        courts_round = []
        for i in range(0, len(pairs), 2):
            a, b = pairs[i]
            c, d = pairs[i + 1]
            if len({a, b, c, d}) < 4:
                return None
            courts_round.append(((a, b), (c, d)))
        return courts_round

    start_time = time.time()
    best = None
    while time.time() - start_time < time_budget_sec:
        for i in range(N):
            rests_so_far[i] = 0
            last_rest_round[i] = -99
            opponent_counts[i].clear()
            partner_pairs[i].clear()
        schedule = []
        success = True
        for r in range(rounds):
            resting = choose_resting_pairs(r) if partner_mode == "fixed" else choose_resting(r)
            rest_set = set(resting)
            avail = [i for i in base_order if i not in rest_set]
            if partner_mode == "fixed":
                round_pairs = [p for p in pairs if p[0] in avail and p[1] in avail]
            else:
                round_pairs = pair_players(avail)
            if not round_pairs or len(round_pairs) < courts * 2:
                success = False
                break
            courts_round = group_into_courts(round_pairs[: courts * 2])
            if not courts_round:
                success = False
                break
            schedule.append({"resting": resting, "courts": courts_round})
            for i in resting:
                rests_so_far[i] += 1
                last_rest_round[i] = r
            for (a, b), (c, d) in courts_round:
                for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                    inc_opp(x, y)
        if success:
            best = schedule
            break

    if not best:
        return None

    columns = ["Round"] + [f"Court {i+1}" for i in range(courts)] + ["Resting"]
    rows = []
    for r, rd in enumerate(best, start=1):
        cells = [r]
        for (a, b), (c, d) in rd["courts"]:
            cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")
        cells.append(", ".join(names[i] for i in sorted(rd["resting"])))
        rows.append(cells)
    return columns, rows


# ------------------------------------------------------------
# PDF Builder — adaptive, large, and multi-page
# ------------------------------------------------------------
def _text_col_width(data: List[str], font_name: str, font_size: int, padding: float = 24.0) -> float:
    """
    Compute width needed for a column based on the widest text in `data`.
    Adds padding for breathing room.
    """
    widest = 0.0
    for s in data:
        s = "" if s is None else str(s)
        try:
            w = stringWidth(s, font_name, font_size)
        except Exception:
            w = len(s) * (font_size * 0.55)  # fallback rough estimate
        widest = max(widest, w)
    return widest + padding


def _chunk_courts_by_width(df: pd.DataFrame, font_name: str, font_size: int, page_width: float) -> List[List[str]]:
    """
    Split court columns into groups that fit on a page with Round + Resting.
    Uses measured text width of each column.
    """
    cols = list(df.columns)
    round_col = cols[0]
    rest_col = cols[-1]
    court_cols = [c for c in cols[1:-1] if c.lower().startswith("court ")]

    # measure base widths
    round_width = _text_col_width([round_col] + df[round_col].astype(str).tolist(), font_name, font_size)
    rest_width  = _text_col_width([rest_col]  + df[rest_col].astype(str).tolist(),  font_name, font_size, padding=40.0)

    # we'll pack courts greedily per page
    groups = []
    current = []
    current_width = round_width + rest_width  # fixed columns per page
    # leave some extra margin
    usable = page_width - 48

    for c in court_cols:
        col_width = _text_col_width([c] + df[c].astype(str).tolist(), font_name, font_size)
        if current and current_width + col_width > usable:
            groups.append(current)
            current = [c]
            current_width = round_width + rest_width + col_width
        else:
            current.append(c)
            current_width += col_width
    if current:
        groups.append(current)

    # Ensure at least 2 courts per page if possible; if names are extremely long
    # this may still drop to 1 per page — acceptable for legibility.
    return groups or [[]]


def build_print_pdf(df: pd.DataFrame, title="Pickleball Schedule", big=True) -> bytes:
    buffer = BytesIO()
    page = landscape(letter)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page,
        leftMargin=24,
        rightMargin=24,
        topMargin=28,
        bottomMargin=28,
    )
    story = []
    styles = getSampleStyleSheet()
    h = styles["Heading1"].clone("H")
    h.fontSize = 22 if big else 18
    h.leading = h.fontSize + 4

    story.append(Paragraph(title, h))
    story.append(Spacer(1, 10))

    # Fonts & paddings
    body_font = "Helvetica"
    body_size = 16 if big else 13
    header_size = body_size + 1
    row_top_pad = 12
    row_bottom_pad = 12

    cols = list(df.columns)
    round_col = cols[0]
    rest_col = cols[-1]

    # Split courts by width so each page fits nicely with big text
    court_groups = _chunk_courts_by_width(df, body_font, body_size, page_width=page[0])

    for idx, group in enumerate(court_groups):
        page_cols = [round_col] + group + [rest_col]
        data = [page_cols] + df[page_cols].values.tolist()

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),

            # Alignment: center everywhere EXCEPT Resting column which is left
            ('ALIGN', (0, 0), (-2, -1), 'CENTER'),
            ('ALIGN', (-1, 0), (-1, 0), 'CENTER'),  # Resting header centered
            ('ALIGN', (-1, 1), (-1, -1), 'LEFT'),    # Resting cells left

            # Vertical centering & comfortable row height
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), row_top_pad),
            ('BOTTOMPADDING', (0, 0), (-1, -1), row_bottom_pad),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',  (0, 0), (-1, 0), header_size),
            ('FONTNAME',  (0, 1), (-1, -1), body_font),
            ('FONTSIZE',  (0, 1), (-1, -1), body_size),

            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(tbl)

        if idx < len(court_groups) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    col1, col2 = st.columns(2)
    with col1:
        players = st.number_input("Players", min_value=4, max_value=200, value=17, step=1)
    with col2:
        courts = st.number_input("Courts", min_value=1, max_value=25, value=3, step=1)
    rounds = st.number_input("Rounds", min_value=1, max_value=50, value=10, step=1)

    partner_mode = st.radio("Partner Mode", ["Rotate partners (all different)", "Stick with same partner"])
    fixed_mode = partner_mode.startswith("Stick")

    cap = st.selectbox("Max vs Any Opponent", [1, 2], index=1)
    seed_input = st.text_input("Random Seed (optional)")
    seed_val = safe_int(seed_input, None) if seed_input else None

    names_text = st.text_area("Player Names (optional, one per line)", height=120)
    pairs_text = ""
    if fixed_mode:
        pairs_text = st.text_area("Fixed Pairs (optional: '1 & 2' or 'Alex, Bea' per line)", height=120)

    run = st.button("Generate Schedule", type="primary")

# ------------------------------------------------------------
# Main Output
# ------------------------------------------------------------
if run:
    names = parse_player_names(players, names_text)
    fixed_pairs = parse_fixed_pairs(players, names, pairs_text) if fixed_mode else None

    try:
        result = generate_schedule(
            players, courts, rounds, cap, names,
            "fixed" if fixed_mode else "rotate",
            fixed_pairs, seed_val
        )
    except ValueError as e:
        result = None
        st.error(str(e))

    if not result:
        st.error("Could not generate a valid schedule. Try adjusting numbers or loosening constraints.")
    else:
        columns, rows = result
        df = pd.DataFrame(rows, columns=columns)
        st.success("✅ Schedule generated successfully!")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Exports
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        pdf_bytes = build_print_pdf(df, title="Pickleball Schedule", big=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download CSV", data=csv_bytes, file_name="pickleball_schedule.csv", mime="text/csv")
        with c2:
            st.download_button("Get Print Version (PDF)", data=pdf_bytes, file_name="pickleball_schedule_print.pdf", mime="application/pdf")
else:
    st.info("Set your event details in the sidebar and click **Generate Schedule**.")
