# Usage examples

All examples use the command-line demo:

```bash
python -m src.cli "your question"
```

The outputs below are the assistant's real responses over the bundled dataset. Exact numbers
depend on `data/nba_dataset.csv`.

## Human-readable answers

```bash
python -m src.cli "How many points do the Warriors average over the last 5 games?"
# Golden State Warriors averaged 114.4 points over the last 5 games.

python -m src.cli "How many points do GSW allow over the last 5 games?"
# Golden State Warriors allowed 117.0 points per game over the last 5 games.

python -m src.cli "What is the Warriors record?"
# Golden State Warriors are 289-223 across 512 games.

python -m src.cli "Top 5 scoring teams"
# Top scoring teams: 1. Atlanta Hawks - 116.13 points per game; 2. ...

python -m src.cli "Celtics vs Heat head to head"
# Boston Celtics are 25-14 against Miami Heat across 39 meetings.

python -m src.cli "Boston Celtics efficiency last 10 games"
# Boston Celtics over the last 10 games: ORTG 106.98, DRTG 101.93, net rating 5.05.

python -m src.cli "How are the Warriors performing over the last 5 games?"
# Golden State Warriors over the last 5 games: 2-3 record, 114.4 points scored per game,
# 117.0 points allowed, 107.04 ORTG, 109.78 DRTG, and -2.73 net rating.

python -m src.cli "What is the Warriors home record?"
# Golden State Warriors are 170-86 across 256 home games.

python -m src.cli "How many points do the Celtics score on the road over the last 5 games?"
# Boston Celtics averaged 104.2 points over the last 5 away games.
```

Note how the validator canonicalises the team: `Warriors` and `GSW` both become
`Golden State Warriors` in the answer.

**Simple vs broad answers.** A single-metric question keeps a simple answer; a broad performance
question (`How are the Warriors performing…`, `advanced profile`, `summarise the Warriors over the
last 10 games`, `compare the Warriors offense and defense over the last 5 games`) returns a fuller
**profile** — record, points scored and allowed, and pace-adjusted ratings. Simple answers are never
auto-enriched with this extra context.

**Home/away splits.** The five single-team families (average points, points allowed, record,
efficiency, advanced profile) — and the two-team comparison — accept an optional `home`/`away` filter
(`at home`, `on the road` also work); with a window, `last N` then means the last N home/away games
(applied per team for a comparison). `top scoring teams` and `head-to-head` do not support location
splits — those queries fail safely with a clear message. This stays offline and deterministic — no
live data, no betting forecasts, and no UI.

**Two-team comparison.** A descriptive side-by-side comparison of two teams over the same sample
type. Phrase it with explicit comparison language — `compare X and Y`, `how do X and Y compare`,
`comparison between X and Y`:

```bash
python -m src.cli "Compare Warriors and Celtics over the last 10 games."
# Over the last 10 games, Golden State Warriors went 4-6 with 110.9 points scored per game, 114.7
# allowed, and a -3.59 net rating. Boston Celtics went 7-3 with 108.5 points scored per game, 100.7
# allowed, and a +5.05 net rating. Boston Celtics had the stronger profile over this selected sample
# based on net rating.

python -m src.cli "Compare Lakers and Knicks at home."
# Across all available home games, Los Angeles Lakers ... New York Knicks ... (per-team home games).
```

It is **descriptive, not predictive**: it compares net rating, record, scoring, and margin over the
*selected sample* and never claims one team will win or is "better overall". When the net-rating gap
is very small the verdict stays neutral. Comparison is distinct from **head-to-head** (which reports
the games two teams played against each other): bare `X vs Y` keeps its existing head-to-head
meaning, and `who is better` / `who will win` / betting questions remain unsupported.

## JSON output

Add `--json` to print the full structured `AssistantResult`:

```bash
python -m src.cli --json "Celtics vs Heat head to head"
```

A compact view of the shape (the real payload also includes `data` and `meta`):

```json
{
  "status": "answer",
  "tool_name": "head_to_head",
  "message": "Boston Celtics are 25-14 against Miami Heat across 39 meetings.",
  "query": "Celtics vs Heat head to head",
  "errors": [],
  "warnings": []
}
```

JSON output is deterministic (`sort_keys=True`) and always valid JSON.

## Optional Rich pretty mode

`--pretty` renders the same `AssistantResult` in a polished terminal layout — a panel for messages,
or a table for `compare_team_profiles` and `top_scoring_teams`:

```bash
python -m src.cli --pretty "Compare Warriors and Celtics over the last 10 games."
# Team comparison
#   Team                    Games   W-L     PPG     OPP    ORTG    DRTG    Net
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   Golden State Warriors      10   4-6   110.9   114.7   105.6   109.2   -3.6
#   Boston Celtics             10   7-3   108.5   100.7   107.0   101.9   +5.0
# (PPG = points scored, OPP = points allowed, Net = net rating)
# ╭─ SUMMARY ─────────────────────────────────────────────────────────────────╮
# │ Boston Celtics had the stronger profile over this selected sample based on │
# │ net rating.                                                                │
# ╰───────────────────────────────────────────────────────────────────────────╯
```

`--pretty` is a **Rich terminal presentation layer only**: it does not change parsing, validation,
analytics, result calculation, exit codes, or JSON output — it just re-renders the existing result.
It needs the optional `rich` dependency (`pip install -r requirements-rich.txt`); without it,
`--pretty` prints an install hint and exits, while the plain and `--json` modes still work.
`--pretty` and `--json` are mutually exclusive.

## Safe failure examples

The assistant explains why it cannot answer; it never guesses a number.

```bash
python -m src.cli "Who is better?"
# I can only answer supported NBA analytics questions, such as team averages, points allowed,
# records, top scoring teams, head-to-head records, efficiency summaries, advanced team profiles,
# and team-to-team comparisons.  (unsupported)

python -m src.cli "How many points do LA average?"
# "LA" is ambiguous. Do you mean Los Angeles Lakers or Los Angeles Clippers?            (clarification_needed)

python -m src.cli "How many points do Celics average?"
# I could not find "Celics". Did you mean Boston Celtics?                                (clarification_needed)
# (a typo yields a single best suggestion — it is offered, never auto-applied)

python -m src.cli "Celtics vs Celtics head to head"
# A head-to-head query needs two different teams.                                       (clarification_needed)

python -m src.cli "Compare Celtics and Celtics."
# A comparison needs two different teams.                                               (clarification_needed)
```

In `--json` mode these carry a structured issue code in `errors`:

| Query | Status | Issue code |
| --- | --- | --- |
| `Who is better?` | `unsupported` | `unsupported_query` |
| `How many points do LA average?` | `clarification_needed` | `ambiguous_team` |
| `How many points do Celics average?` | `clarification_needed` | `unknown_team` |
| `Celtics vs Celtics head to head` | `clarification_needed` | `same_team_head_to_head` |
| `Compare Celtics and Celtics.` | `clarification_needed` | `same_team_comparison` |

`LA` is ambiguous (Lakers or Clippers), `Celics` is an unrecognised spelling, and a team cannot
play a head-to-head against itself — each is reported precisely rather than guessed.

## Exit codes

The CLI returns a deterministic exit code so it can be scripted:

| Exit code | Meaning |
| --- | --- |
| `0` | `answer` — the question was answered |
| `1` | `clarification_needed` or `unsupported` — a safe, explained non-answer |
| `2` | assistant `error`, runtime/bootstrap failure, or invalid command-line arguments |

```bash
python -m src.cli "Who is better?"; echo $?   # prints the message, then 1
```

## Tips

- Teams can be a nickname (`Warriors`), tri-code (`GSW`), full name (`Boston Celtics`), or an
  unambiguous city (`Boston`). Ambiguous markets (`LA`, `NY`) ask for clarification.
- Add a window with `last N games` (or `last N meetings` for head-to-head). Omitting it uses all
  available games.
- Vague time phrases (`recently`, `last few games`) are intentionally rejected rather than guessed.
