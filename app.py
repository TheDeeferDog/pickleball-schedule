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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
st.caption(
    "Rotate partners, fixed team pairs, evenly spaced rests, capped repeat opponents, "
    "and printable large-font schedules."
)


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
    return [lines[i] if i < len(lines) else f"{i + 1}" for i in range(count)]


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


def parse_team_pairs(text: str) -> List[Tuple[str, str]]:
    """
    Parses fixed team pairs entered one per line.

    Accepted formats:
    Beddie & John
    Beddie, John
    Beddie / John
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    pairs = []

    for ln in lines:
        normalized = ln.replace("&", ",").replace("/", ",")
        parts = [p.strip() for p in normalized.split(",") if p.strip()]

        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))

    return pairs


# ------------------------------------------------------------
# Fixed-pair round robin generator
# ------------------------------------------------------------
def generate_fixed_pair_round_robin(team_pairs: List[Tuple[str, str]], courts: int):
    """
    Generates a fixed-pair round robin schedule.

    Each pair stays together.
    Each pair plays every other pair once.
    Games are grouped into schedule rows based on number of courts.
    """

    if len(team_pairs) < 2:
        return None

    teams = [f"{a} & {b}" for a, b in team_pairs]

    # If odd number of teams, add a BYE placeholder.
    if len(teams) % 2 == 1:
        teams.append("BYE")

    n = len(teams)
    team_order = teams[:]
    round_groups = []

    # Circle method round robin.
    for _ in range(n - 1):
        games_this_round = []

        for i in range(n // 2):
            team_a = team_order[i]
            team_b = team_order[n - 1 - i]

            if team_a != "BYE" and team_b != "BYE":
                games_this_round.append((team_a, team_b))

        round_groups.append(games_this_round)

        # Rotate all teams except the first.
        team_order = [team_order[0]] + [team_order[-1]] + team_order[1:-1]

    columns = ["Round"] + [f"Court {i + 1}" for i in range(courts)] + ["Resting"]
    rows = []

    display_round = 1
    real_teams = [team for team in teams if team != "BYE"]

    for rr_round_games in round_groups:
        # Split round-robin games into court-sized blocks.
        for i in range(0, len(rr_round_games), courts):
            block = rr_round_games[i:i + courts]

            active_teams = set()
            cells = [display_round]

            for team_a, team_b in block:
                cells.append(f"{team_a} V {team_b}")
                active_teams.add(team_a)
                active_teams.add(team_b)

            # Fill unused courts.
            while len(cells) < courts + 1:
                cells.append("")

            resting = [team for team in real_teams if team not in active_teams]
            cells.append(", ".join(resting))

            rows.append(cells)
            display_round += 1

    return columns, rows


# ------------------------------------------------------------
# Schedule generator for rotating partners
# ------------------------------------------------------------
def generate_schedule(
    N,
    courts,
    rounds,
    max_opp_repeat,
    names,
    partner_mode,
    fixed_pairs=None,
    seed=None,
    time_budget_sec=4.0,
):
    if seed is not None:
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
        # Build pairs trying to avoid repeat partners in rotate mode.
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

    def courts_ok_with_cap(courts_round):
        # Ensure adding this set of courts does not exceed max_opp_repeat for any matchup.
        for (a, b), (c, d) in courts_round:
            for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                if opponent_counts[x][y] + 1 > max_opp_repeat:
                    return False

        return True

    def group_into_courts_with_cap(pairs_for_round):
        # Try many shuffles to find a grouping that respects the opponent cap.
        idxs = list(range(len(pairs_for_round)))

        for _ in range(1500):
            random.shuffle(idxs)
            ok = True
            tmp = []

            for i in range(0, len(idxs), 2):
                a, b = pairs_for_round[idxs[i]]
                c, d = pairs_for_round[idxs[i + 1]]

                if len({a, b, c, d}) < 4:
                    ok = False
                    break

                tmp.append(((a, b), (c, d)))

            if not ok:
                continue

            if courts_ok_with_cap(tmp[:courts]):
                return tmp[:courts]

        return None

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

            courts_round = group_into_courts_with_cap(round_pairs[: courts * 2])

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

    columns = ["Round"] + [f"Court {i + 1}" for i in range(courts)] + ["Resting"]
    rows = []

    for r, rd in enumerate(best, start=1):
        cells = [r]

        for (a, b), (c, d) in rd["courts"]:
            cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")

        cells.append(", ".join(names[i] for i in sorted(rd["resting"])))
        rows.append(cells)

    return columns, rows


# ------------------------------------------------------------
# PDF Builder — narrow Round, Resting treated like a court, adaptive paging
# ------------------------------------------------------------
def _measure_width(texts: List[str], font: str, size: int, pad: float) -> float:
    widest = 0.0

    for s in texts:
        s = "" if s is None else str(s)

        try:
            w = stringWidth(s, font, size)
        except Exception:
            w = len(s) * (size * 0.55)

        widest = max(widest, w)

    return widest + pad


def _group_cols_across_pages(
    df: pd.DataFrame,
    font: str,
    size: int,
    page_width: float,
    margins: Tuple[float, float],
) -> List[List[str]]:
    """
    Pack variable columns, Court 1..N and Resting, across pages alongside
    a fixed narrow Round column.

    Resting is treated like a court and may appear only on the final page(s).
    """
    left_margin, right_margin = margins
    cols = list(df.columns)

    round_col = cols[0]
    court_cols = [c for c in cols[1:-1] if c.lower().startswith("court ")]
    resting_col = cols[-1]
    var_cols = court_cols + [resting_col]

    round_samples = [round_col] + [str(i) for i in range(1, min(100, len(df) + 1))]
    round_w = _measure_width(round_samples, font, size, pad=18)

    usable = page_width - (left_margin + right_margin)

    groups = []
    cur = []
    cur_width = round_w

    for c in var_cols:
        col_w = _measure_width([c] + df[c].astype(str).tolist(), font, size, pad=30)

        if cur and cur_width + col_w > usable:
            groups.append(cur)
            cur = [c]
            cur_width = round_w + col_w
        else:
            cur.append(c)
            cur_width += col_w

    if cur:
        groups.append(cur)

    capped = []

    for g in groups:
        if len(g) <= 4:
            capped.append(g)
        else:
            for i in range(0, len(g), 4):
                capped.append(g[i:i + 4])

    return capped or [[]]


def build_print_pdf(df: pd.DataFrame, title="Pickleball Schedule", pdf_size="Large") -> bytes:
    buffer = BytesIO()
    page = landscape(letter)

    left_margin = right_margin = top_margin = bottom_margin = 36

    doc = SimpleDocTemplate(
        buffer,
        pagesize=page,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=title,
    )

    story = []

    styles = getSampleStyleSheet()
    header_style = styles["Heading1"].clone("SchedHeader")

    # Font sizing.
    if pdf_size == "X-Large":
        body_size = 18
        header_style.fontSize = 24
    else:
        body_size = 16
        header_style.fontSize = 22

    header_style.leading = header_style.fontSize + 6

    body_font = "Helvetica"
    body_leading = body_size + 4

    cell_center = ParagraphStyle(
        "CellCenter",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=body_size,
        leading=body_leading,
        alignment=1,
    )

    cell_left = ParagraphStyle(
        "CellLeft",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=body_size,
        leading=body_leading,
        alignment=0,
    )

    row_top_pad = 12
    row_bottom_pad = 12
    left_pad = 10
    right_pad = 10

    cols = list(df.columns)
    round_col = cols[0]
    resting_col = cols[-1]

    groups = _group_cols_across_pages(
        df,
        body_font,
        body_size,
        page_width=page[0],
        margins=(left_margin, right_margin),
    )

    for idx, group in enumerate(groups):
        courts_in_group = [c for c in group if c != resting_col]
        show_resting = resting_col in group

        if courts_in_group:
            first_c = courts_in_group[0].split()[-1]
            last_c = courts_in_group[-1].split()[-1]
            subtitle = (
                f"{title} — Court {first_c}"
                if first_c == last_c
                else f"{title} — Courts {first_c}–{last_c}"
            )
        else:
            subtitle = f"{title} — Resting"

        story.append(Paragraph(subtitle, header_style))
        story.append(Spacer(1, 10))

        page_cols = [round_col] + group
        data = [page_cols]

        for _, row in df.iterrows():
            row_cells = []

            for c in page_cols:
                txt = "" if pd.isna(row[c]) else str(row[c])

                if c == resting_col:
                    row_cells.append(Paragraph(txt, cell_left))
                else:
                    row_cells.append(Paragraph(txt, cell_center))

            data.append(row_cells)

        # Widths: Round narrow measured; others scaled.
        round_samples = [round_col] + [str(i) for i in range(1, min(100, len(df) + 1))]
        round_w = _measure_width(round_samples, body_font, body_size, pad=18)

        avail_w = page[0] - (left_margin + right_margin)
        var_ws = []
        total_var = 0.0

        for c in group:
            pad_amt = 36 if c == resting_col else 30
            w = _measure_width([c] + df[c].astype(str).tolist(), body_font, body_size, pad=pad_amt)
            var_ws.append(w)
            total_var += w

        remaining = max(avail_w - round_w, 200)
        scale = min(1.0, remaining / total_var) if total_var > 0 else 1.0
        col_widths = [round_w] + [w * scale for w in var_ws]

        tbl = Table(data, repeatRows=1, colWidths=col_widths, hAlign="CENTER")

        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), row_top_pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), row_bottom_pad),
            ("LEFTPADDING", (0, 0), (-1, -1), left_pad),
            ("RIGHTPADDING", (0, 0), (-1, -1), right_pad),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), body_size + 1),
        ]

        if show_resting:
            table_style.append(("ALIGN", (-1, 1), (-1, -1), "LEFT"))

        tbl.setStyle(TableStyle(table_style))
        story.append(tbl)

        if idx < len(groups) - 1:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    partner_mode = st.radio(
        "Schedule Mode",
        ["Rotate partners", "Fixed team pairs"],
        help=(
            "Rotate partners creates individual-player schedules. "
            "Fixed team pairs keeps each pair together and schedules pair-vs-pair games."
        ),
    )

    fixed_mode = partner_mode == "Fixed team pairs"

    col1, col2 = st.columns(2)

    if fixed_mode:
        with col1:
            courts = st.number_input("Courts", min_value=1, max_value=25, value=2, step=1)

        with col2:
            pdf_size = st.radio("PDF Font Size", ["Large", "X-Large"], index=0)

        pairs_text = st.text_area(
            "Fixed Team Pairs",
            value="Beddie & John\nAlex & Maria\nSam & Lee\nPat & Chris",
            help="Enter one pair per line. Example: Beddie & John",
            height=180,
        )

        players = None
        rounds = None
        cap = None
        seed_val = None
        names_text = ""

    else:
        with col1:
            players = st.number_input("Players", min_value=4, max_value=200, value=17, step=1)

        with col2:
            courts = st.number_input("Courts", min_value=1, max_value=25, value=3, step=1)

        rounds = st.number_input("Rounds", min_value=1, max_value=50, value=10, step=1)

        cap = st.selectbox("Max vs Any Opponent", [1, 2, 3, 4], index=1)
        pdf_size = st.radio("PDF Font Size", ["Large", "X-Large"], index=0)

        seed_input = st.text_input(
            "Specific Schedule Number",
            help=(
                "Enter a number to reproduce the exact same schedule later. "
                "Leave blank for a new random draw each time."
            ),
        )

        seed_val = safe_int(seed_input, None) if seed_input else None

        names_text = st.text_area("Player Names, optional, one per line", height=120)
        pairs_text = ""

    run = st.button("Generate Schedule", type="primary")


# ------------------------------------------------------------
# Main Output
# ------------------------------------------------------------
if run:
    try:
        if fixed_mode:
            team_pairs = parse_team_pairs(pairs_text)

            if len(team_pairs) < 2:
                result = None
                st.error("Please enter at least two fixed pairs.")
            else:
                result = generate_fixed_pair_round_robin(team_pairs, courts)

        else:
            names = parse_player_names(players, names_text)

            result = generate_schedule(
                players,
                courts,
                rounds,
                cap,
                names,
                "rotate",
                None,
                seed_val,
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

        # Exports.
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        pdf_bytes = build_print_pdf(df, title="Pickleball Schedule", pdf_size=pdf_size)

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="pickleball_schedule.csv",
                mime="text/csv",
            )

        with c2:
            st.download_button(
                "Get Print Version PDF",
                data=pdf_bytes,
                file_name="pickleball_schedule_print.pdf",
                mime="application/pdf",
            )

else:
    st.info("Set your event details in the sidebar and click **Generate Schedule**.")
