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
efficiency, advanced profile) accept an optional `home`/`away` filter (`at home`, `on the road` also
work); with a window, `last N` then means the last N home/away games. `top scoring teams` and
`head-to-head` do not support location splits — those queries fail safely with a clear message. This
stays offline and deterministic — no live data, no betting forecasts, and no UI.

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

## Safe failure examples

The assistant explains why it cannot answer; it never guesses a number.

```bash
python -m src.cli "Who is better?"
# I can only answer supported NBA analytics questions, such as team averages, points allowed,
# records, top scoring teams, head-to-head records, efficiency summaries, and advanced team
# profiles.  (unsupported)

python -m src.cli "How many points do LA average?"
# "LA" is ambiguous. Do you mean Los Angeles Lakers or Los Angeles Clippers?            (clarification_needed)

python -m src.cli "How many points do Celics average?"
# I could not find "Celics". Did you mean Boston Celtics?                                (clarification_needed)
# (a typo yields a single best suggestion — it is offered, never auto-applied)

python -m src.cli "Celtics vs Celtics head to head"
# A head-to-head query needs two different teams.                                       (clarification_needed)
```

In `--json` mode these carry a structured issue code in `errors`:

| Query | Status | Issue code |
| --- | --- | --- |
| `Who is better?` | `unsupported` | `unsupported_query` |
| `How many points do LA average?` | `clarification_needed` | `ambiguous_team` |
| `How many points do Celics average?` | `clarification_needed` | `unknown_team` |
| `Celtics vs Celtics head to head` | `clarification_needed` | `same_team_head_to_head` |

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
