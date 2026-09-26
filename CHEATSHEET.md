# F1 Winner Predictor — Weekend Cheat Sheet

## Where we are

- [x] Step 1 — Mental model (5 building blocks + MLflow)
- [x] Step 2 — Project skeleton, uv environment (12 packages installed)
- [x] Step 3 — Data pipeline: 6,911 rows (2010–2025) from Jolpica API → `data/races.csv`
- [ ] Step 4 — Training script (next)
- [ ] Step 5 — MLflow experiment tracking
- [ ] Step 6 — FastAPI inference service
- [ ] Step 7 — Tests
- [ ] Step 8 — Docker
- [ ] Step 9 — GitHub Actions CI/CD
- [ ] Step 10 — Cloud deploy (Render/Railway)
- [ ] Step 11 — README + CV story

## Warm-up (first thing Saturday)

```bash
cd ~/projects/f1-winner-predictor
source .venv/bin/activate        # prompt shows (f1-winner-predictor)
git log --oneline                # 2 commits so far
git status                       # should be clean
python src/fetch_data.py         # only if data/races.csv is missing (idempotent, overwrites)
```

## Project layout

```
app/               FastAPI service code          (Step 6)
src/               fetch_data.py, train.py       (Steps 3–5)
data/races.csv     6,911 rows of F1 results      (generated, committed)
tests/             test files                    (Step 7)
model/             trained model artifact        (Step 4)
.github/workflows/ CI/CD pipeline                (Step 9)
```

## Commands that matter

| Task | Command |
|---|---|
| See where you are | `pwd` |
| Activate env | `source .venv/bin/activate` |
| Install a package | `uv pip install "name==x.y.z"` |
| Freeze deps (for Docker/CI later) | `uv pip freeze > requirements.txt` |
| Snapshot work | `git add -A && git commit -m "message"` |
| Inspect any API response | `curl -s "<url>" \| python -m json.tool \| head -80` |
| Check DNS for a domain | `nslookup <domain> 8.8.8.8` |
| Find an env var's source | `env \| grep -i <name>` |

## Concepts bank (say these in interviews)

- **Data leakage** — never feed the model anything unknowable before the race (e.g., final position). Features must exist at prediction time.
- **position vs positionOrder** — `position` is missing for DNFs; `positionOrder` is always numeric. Target = `position_order == 1`. Chose the field that never lies.
- **Fail loudly** — `raise_for_status()` crashes on bad HTTP instead of writing garbage.
- **Idempotency** — re-running the fetcher gives the same CSV (`open(..., "w")` truncates). `"a"` would append = classic pipeline bug.
- **Virtual environment** — isolated package pile per project; uv makes venvs and fetches its own Pythons.
- **Docker** — solves "it works on my machine": code + model + exact deps in one portable image; runs identically anywhere.
- **CI/CD** — push → tests run → green ships, red blocks. Tests failing never touch production.
- **Read tracebacks bottom-up** — the deepest line is where the failure began.
- **NXDOMAIN** = "this domain doesn't exist" (different from network down!). Suspect the *name* first.

## Battle scars from Friday (do not re-live these)

1. **`UV_PROJECT` env var** — sets a global redirect for uv; we `unset` it in session. Still TODO: find source (`~/.config/envman/load.sh` was clean; next suspects: `~/.local/bin/env`). Until removed, open terminals may redirect `uv venv` to the Anki folder!
2. **Typo: `api.jolpica.ca` ≠ `api.jolpi.ca`** — the extra "a" = NXDOMAIN. Correct BASE_URL: `https://api.jolpi.ca/ergast/f1`.
3. **Indentation bug** — pagination block must sit OUTSIDE the `for result` loop (once per page), not inside (once per driver). Friday's fingerprint: exactly 9 rows/season. Healthy range: ~300–600/season.
4. **`"w"` vs `"a"`** — write mode recreates the CSV; append would duplicate rows on re-runs.
5. **Conda `(base)` vs `.venv`** — leave base alone; the project venv stacks on top.

## Saturday preview

- **Train/test split by TIME** (e.g., train ≤ 2023, test 2024–2025) — never randomly; random splits leak the future into training. That's the #1 credibility point of this project.
- **Baseline first**: pole-sitter win rate (run `(df[df.grid==1].position_order == 1).mean()` tonight if curious — every model must beat this number).
- **Two candidates**: RandomForestClassifier vs GradientBoostingClassifier (LogisticRegression as a sanity baseline). Compare accuracy + log-loss.
- **Leakage rule for features**: only pre-race info — grid, constructor strength (historical points), circuit, driver form.

## Key fields in data/races.csv

`season, round, race_name, circuit, date, driver_code, driver_id, constructor, grid, position, position_order, points, laps, status`

- Winner: `position_order == 1`
- `position` = official classification (None for unclassified); `position_order` = always-present finishing order
- `status`: "Finished" / "+1 Lap" / DNF reasons — useful for a DNF feature later

## Friday night final state

```
2 commits, 6,911 rows, env working, API verified.
Everything above this line is proven. Everything below is Saturday.
```
