import time
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Optional
from io import BytesIO
from itertools import permutations

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
APP_VERSION = "balanced court assignment v6 - diagnostics"

st.set_page_config(
    page_title="Pickleball Schedule Generator",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Pickleball Schedule Generator")
st.caption(
    f"Version: {APP_VERSION} — Rotate partners, fixed team pairs, named courts, "
    "evenly spaced rests, capped repeat opponents, balanced court assignments, "
    "printable large-font schedules, and helpful failure diagnostics."
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


def diagnose_schedule_inputs(
    N: int,
    courts: int,
    rounds: int,
    max_opp_repeat: int,
) -> Tuple[List[str], List[str]]:
    """
    Returns two lists:
    - blocking_errors: problems that make generation definitely impossible
    - advisory_notes: warnings or explanations that may help if generation fails

    These diagnostics do not replace the generator. They explain the most
    common failure causes before or after generation is attempted.
    """
    blocking_errors = []
    advisory_notes = []

    on_court = courts * 4

    if N < 4:
        blocking_errors.append(
            "You need at least 4 players for one doubles game."
        )

    if courts < 1:
        blocking_errors.append(
            "You need at least 1 court."
        )

    if rounds < 1:
        blocking_errors.append(
            "You need at least 1 round."
        )

    if N < on_court:
        blocking_errors.append(
            f"You selected {courts} court(s), which requires {on_court} players "
            f"per round, but only {N} player(s) are available. Reduce courts or add players."
        )

    if N <= 0:
        return blocking_errors, advisory_notes

    target_rests, rest_per_round = compute_target_rests(N, rounds, courts)

    if N == on_court and rounds > 1:
        advisory_notes.append(
            "Everyone plays every round, so there are no rests to rotate. "
            "That is allowed, but repeated opponents may become harder to avoid."
        )

    # Partner uniqueness check.
    # The current rotating-partner generator avoids repeat partners completely.
    # Each player can have at most N - 1 unique partners.
    if target_rests:
        max_games_for_any_player = rounds - min(target_rests)

        if max_games_for_any_player > N - 1:
            advisory_notes.append(
                f"Some players may need to play {max_games_for_any_player} games, "
                f"but each player only has {N - 1} possible unique partners. "
                "Because this generator avoids repeat partners, this setup may be impossible. "
                "Try fewer rounds, more players, or add a future setting that allows repeat partners."
            )

    # Opponent cap feasibility estimate.
    # In each doubles game, every player faces 2 opponents.
    # Across all games, total directed player-opponent exposures = active player slots * 2.
    total_active_slots = rounds * courts * 4
    total_opponent_exposures = total_active_slots * 2

    # Each unordered player pair can appear as opponents up to max_opp_repeat times.
    # Since total_opponent_exposures is directed, multiply unordered capacity by 2.
    max_opponent_capacity = (N * (N - 1) // 2) * max_opp_repeat * 2

    if max_opponent_capacity > 0 and total_opponent_exposures > max_opponent_capacity:
        advisory_notes.append(
            f"The opponent cap may be too strict. With {N} players, {rounds} rounds, "
            f"and {courts} court(s), the schedule needs about "
            f"{total_opponent_exposures} player-opponent exposures, but the selected "
            f"cap allows about {max_opponent_capacity}. "
            "Try increasing 'Max vs Any Opponent' or reducing rounds."
        )

    # Rest fairness note.
    if rest_per_round > 0:
        total_rests = rest_per_round * rounds

        if total_rests < N:
            advisory_notes.append(
                f"Only {total_rests} total rest slot(s) exist across the schedule, "
                f"but there are {N} players. Some players will not rest, which is expected "
                "for this setup."
            )

    return blocking_errors, advisory_notes


def parse_player_names(count: int, text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return [lines[i] if i < len(lines) else f"{i + 1}" for i in range(count)]


def parse_court_names(courts: int, text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    court_names = []

    for i in range(courts):
        if i < len(lines):
            label = lines[i].strip()
            if label.lower().startswith("court") or "court" in label.lower():
                court_names.append(label)
            else:
                court_names.append(f"Court {label}")
        else:
            court_names.append(f"Court {i + 1}")

    return court_names


def safe_int(s: str, default: Optional[int]) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return default


def split_name_line_without_separator(line: str) -> Optional[Tuple[str, str]]:
    """
    Best-effort split when the user does not type &, comma, slash, or dash.

    Handles examples like:
    Tracy Thompson Dave Jones -> Tracy Thompson & Dave Jones
    Maureen MacLean Peter MacLean -> Maureen MacLean & Peter MacLean
    Michele Van Grol Jeff Solway -> Michele Van Grol & Jeff Solway
    Alfie Colombo Mark Slater -> Alfie Colombo & Mark Slater
    """
    clean = " ".join(line.replace("\t", " ").split())
    words = clean.split()
    n = len(words)

    if n < 2:
        return None

    if n == 2:
        return words[0], words[1]

    if n == 3:
        return words[0], " ".join(words[1:])

    if n == 4:
        return " ".join(words[:2]), " ".join(words[2:])

    if n == 5:
        return " ".join(words[:3]), " ".join(words[3:])

    if n == 6:
        return " ".join(words[:3]), " ".join(words[3:])

    return None


def parse_team_pairs(text: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Parses fixed team pairs entered one per line.

    Accepted examples:
    Beddie & John
    Beddie, John
    Beddie / John
    Beddie - John
    Tracy Thompson Dave Jones
    Michele Van Grol Jeff Solway
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    pairs = []
    skipped_lines = []

    for ln in lines:
        clean = " ".join(ln.replace("\t", " ").split())
        has_separator = any(sep in clean for sep in ["&", ",", "/", " - ", " – ", " — "])

        if has_separator:
            normalized = (
                clean.replace(" & ", ",")
                .replace("&", ",")
                .replace("/", ",")
                .replace(" - ", ",")
                .replace(" – ", ",")
                .replace(" — ", ",")
            )

            parts = [p.strip() for p in normalized.split(",") if p.strip()]

            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
            else:
                skipped_lines.append(ln)

        else:
            split_pair = split_name_line_without_separator(clean)

            if split_pair:
                pairs.append(split_pair)
            else:
                skipped_lines.append(ln)

    return pairs, skipped_lines


def fixed_game_participants(game):
    """
    Fixed team-pair game:
    game = ("Team A", "Team B")
    """
    return [game[0], game[1]]


def rotate_game_participants(game):
    """
    Rotating-player game:
    game = ((a, b), (c, d))
    """
    (a, b), (c, d) = game
    return [a, b, c, d]


def balanced_assign_games_to_courts(games, courts: int, court_counts, participant_func):
    """
    Assigns games to court columns while minimizing repeated court use.

    This does not change:
    - who partners with whom
    - who plays whom
    - who rests
    - opponent caps
    - round-robin structure

    It only chooses which court column each already-valid game appears under.
    """
    court_slots = list(range(courts))
    num_games = len(games)

    if num_games == 0:
        return [None] * courts

    # For typical pickleball use, courts are small enough to evaluate every assignment.
    # For very large court counts, use random sampling to avoid excessive permutations.
    possible_assignments = []

    if courts <= 8 and num_games <= courts:
        possible_assignments = list(permutations(court_slots, num_games))
    else:
        seen = set()
        attempts = min(3000, max(300, courts * 150))

        for _ in range(attempts):
            assignment = tuple(random.sample(court_slots, num_games))
            if assignment not in seen:
                seen.add(assignment)
                possible_assignments.append(assignment)

    best_assignment = None
    best_score = None

    for assignment in possible_assignments:
        score = 0

        for game_index, court_index in enumerate(assignment):
            participants = participant_func(games[game_index])

            for participant in participants:
                current_count = court_counts[participant][court_index]

                # Strongly penalize assigning someone to a court they have already used often.
                score += current_count * 100

                # Slightly prefer courts the participant has never used.
                if current_count == 0:
                    score -= 5

        # Tiny random tie-breaker so equally good options do not always choose the same layout.
        score += random.random()

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    court_games = [None] * courts

    for game_index, court_index in enumerate(best_assignment):
        game = games[game_index]
        court_games[court_index] = game

        for participant in participant_func(game):
            court_counts[participant][court_index] += 1

    return court_games


# ------------------------------------------------------------
# Fixed-pair round robin generator
# ------------------------------------------------------------
def generate_fixed_pair_round_robin(
    team_pairs: List[Tuple[str, str]],
    courts: int,
    court_names: List[str],
):
    """
    Generates a fixed-pair round robin schedule.

    Each pair stays together.
    Each pair plays every other pair once.
    Games are grouped into schedule rows based on number of courts.

    Court assignment is balanced so teams are not repeatedly placed on the same court.
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

    columns = ["Round"] + court_names + ["Resting"]
    rows = []

    display_round = 1
    real_teams = [team for team in teams if team != "BYE"]

    # Tracks each team's court usage.
    court_counts = defaultdict(lambda: [0] * courts)

    for rr_round_games in round_groups:
        for i in range(0, len(rr_round_games), courts):
            block = rr_round_games[i:i + courts]

            # Assign these games to courts in the most balanced way.
            court_games = balanced_assign_games_to_courts(
                block,
                courts,
                court_counts,
                fixed_game_participants,
            )

            active_teams = set()
            cells = [display_round]

            for game in court_games:
                if game is None:
                    cells.append("")
                else:
                    team_a, team_b = game
                    cells.append(f"{team_a} V {team_b}")
                    active_teams.add(team_a)
                    active_teams.add(team_b)

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
    court_names,
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
    base_order = list(range(N))

    # Tracks each player's court usage in rotating mode.
    court_counts = defaultdict(lambda: [0] * courts)

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

    def courts_ok_with_cap(courts_round):
        for (a, b), (c, d) in courts_round:
            for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                if opponent_counts[x][y] + 1 > max_opp_repeat:
                    return False

        return True

    def group_into_courts_with_cap(pairs_for_round):
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

        court_counts = defaultdict(lambda: [0] * courts)
        schedule = []
        success = True

        for r in range(rounds):
            resting = choose_resting(r)
            rest_set = set(resting)
            avail = [i for i in base_order if i not in rest_set]

            round_pairs = pair_players(avail)

            if not round_pairs or len(round_pairs) < courts * 2:
                success = False
                break

            courts_round = group_into_courts_with_cap(round_pairs[: courts * 2])

            if not courts_round:
                success = False
                break

            # Assign valid games to courts in the most balanced way.
            court_games = balanced_assign_games_to_courts(
                courts_round,
                courts,
                court_counts,
                rotate_game_participants,
            )

            courts_round_balanced = [game for game in court_games if game is not None]

            schedule.append({"resting": resting, "court_games": court_games, "courts": courts_round_balanced})

            for i in resting:
                rests_so_far[i] += 1
                last_rest_round[i] = r

            for (a, b), (c, d) in courts_round_balanced:
                for x, y in [(a, c), (a, d), (b, c), (b, d)]:
                    inc_opp(x, y)

        if success:
            best = schedule
            break

    if not best:
        return None

    columns = ["Round"] + court_names + ["Resting"]
    rows = []

    for r, rd in enumerate(best, start=1):
        cells = [r]

        for game in rd["court_games"]:
            if game is None:
                cells.append("")
            else:
                (a, b), (c, d) = game
                cells.append(f"{names[a]} & {names[b]} V {names[c]} & {names[d]}")

        cells.append(", ".join(names[i] for i in sorted(rd["resting"])))
        rows.append(cells)

    return columns, rows


def generate_schedule_with_diagnostics(
    N,
    courts,
    rounds,
    max_opp_repeat,
    names,
    court_names,
    seed=None,
    time_budget_sec=4.0,
):
    """
    Runs input diagnostics, then attempts rotating-partner schedule generation.

    Returns:
        result: schedule result or None
        blocking_errors: list of errors that make generation impossible
        advisory_notes: list of helpful notes or likely failure causes
    """
    blocking_errors, advisory_notes = diagnose_schedule_inputs(
        N=N,
        courts=courts,
        rounds=rounds,
        max_opp_repeat=max_opp_repeat,
    )

    if blocking_errors:
        return None, blocking_errors, advisory_notes

    result = generate_schedule(
        N=N,
        courts=courts,
        rounds=rounds,
        max_opp_repeat=max_opp_repeat,
        names=names,
        court_names=court_names,
        seed=seed,
        time_budget_sec=time_budget_sec,
    )

    if result is None:
        advisory_notes.append(
            "The generator could not find a valid schedule within the search time. "
            "This usually means the constraints are too tight for the selected players, "
            "courts, and rounds."
        )

        advisory_notes.append(
            "Most likely fixes: increase 'Max vs Any Opponent', reduce the number of rounds, "
            "reduce the number of courts, add more players, or try a different Specific Schedule Number."
        )

    return result, blocking_errors, advisory_notes


# ------------------------------------------------------------
# PDF Builder
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
    left_margin, right_margin = margins
    cols = list(df.columns)

    round_col = cols[0]
    resting_col = cols[-1]
    game_cols = cols[1:-1]
    var_cols = game_cols + [resting_col]

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
        game_cols_in_group = [c for c in group if c != resting_col]
        show_resting = resting_col in group

        if game_cols_in_group:
            if len(game_cols_in_group) == 1:
                subtitle = f"{title} — {game_cols_in_group[0]}"
            else:
                subtitle = f"{title} — {game_cols_in_group[0]} to {game_cols_in_group[-1]}"
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
            courts = st.number_input("Courts", min_value=1, max_value=25, value=3, step=1)

        with col2:
            pdf_size = st.radio("PDF Font Size", ["Large", "X-Large"], index=0)

        court_names_text = st.text_area(
            "Court Names, optional",
            value="\n".join(str(i + 1) for i in range(courts)),
            help="Enter one court name per line. Examples: 1, 2, 3 or 2, 4, 5.",
            height=90,
        )

        court_names = parse_court_names(courts, court_names_text)

        pairs_text = st.text_area(
            "Fixed Team Pairs",
            value="",
            placeholder=(
                "Enter one pair per line, for example:\n"
                "Beddie & John\n"
                "Tracy Thompson Dave Jones\n"
                "Michele Van Grol Jeff Solway\n"
                "Sam / Lee"
            ),
            help=(
                "Enter one pair per line. You can use &, comma, slash, dash, "
                "or names without separators."
            ),
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

        court_names_text = st.text_area(
            "Court Names, optional",
            value="\n".join(str(i + 1) for i in range(courts)),
            help="Enter one court name per line. Examples: 1, 2, 3 or 2, 4, 5.",
            height=90,
        )

        court_names = parse_court_names(courts, court_names_text)

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
        result = None
        blocking_errors = []
        advisory_notes = []

        if fixed_mode:
            team_pairs, skipped_lines = parse_team_pairs(pairs_text)

            if skipped_lines:
                st.warning(
                    "Some lines could not be read as pairs and were skipped: "
                    + "; ".join(skipped_lines)
                )

            if team_pairs:
                with st.expander("Parsed fixed teams"):
                    parsed_df = pd.DataFrame(
                        team_pairs,
                        columns=["Player 1", "Player 2"],
                    )
                    st.dataframe(parsed_df, use_container_width=True, hide_index=True)

            if len(team_pairs) < 2:
                result = None
                blocking_errors.append("Please enter at least two fixed pairs.")
            else:
                result = generate_fixed_pair_round_robin(team_pairs, courts, court_names)

                if result is None:
                    advisory_notes.append(
                        "The fixed-pair round robin could not be generated. Check that at least two valid pairs were entered."
                    )

        else:
            names = parse_player_names(players, names_text)

            result, blocking_errors, advisory_notes = generate_schedule_with_diagnostics(
                N=players,
                courts=courts,
                rounds=rounds,
                max_opp_repeat=cap,
                names=names,
                court_names=court_names,
                seed=seed_val,
            )

    except ValueError as e:
        result = None
        blocking_errors = [str(e)]
        advisory_notes = []

    if not result:
        st.error("Could not generate a valid schedule.")

        if blocking_errors:
            st.subheader("What needs to be fixed")
            for msg in blocking_errors:
                st.error(msg)

        if advisory_notes:
            st.subheader("Why this may have failed")
            for msg in advisory_notes:
                st.warning(msg)

        if not blocking_errors and not advisory_notes:
            st.warning(
                "Try adjusting the number of players, courts, rounds, or opponent repeat limit."
            )

    else:
        columns, rows = result
        df = pd.DataFrame(rows, columns=columns)

        st.success("✅ Schedule generated successfully!")

        if advisory_notes:
            with st.expander("Schedule notes"):
                for msg in advisory_notes:
                    st.info(msg)

        st.dataframe(df, use_container_width=True, hide_index=True)

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
