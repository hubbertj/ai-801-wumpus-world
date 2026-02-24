# Wumpus World AI Simulation

A Python simulation of the classic **Wumpus World** AI problem from Russell & Norvig's *Artificial Intelligence: A Modern Approach*.

A logical knowledge-based agent explores a dark cave, reasons about dangers it cannot see, collects gold, and tries to escape alive — all explained step by step in real time.

---

## Features

| Feature | Details |
|---|---|
| **3 difficulty levels** | Easy (4×4), Medium (6×6), Hard (8×8) |
| **Logical AI agent** | Propositional inference, constraint propagation |
| **Visible knowledge base** | Live map of what the agent knows vs. actual world |
| **AI reasoning log** | Every deduction and decision explained in plain English |
| **Pygame 2D display** | Fog-of-war world view + knowledge base map + sidebar |
| **UV project** | One-command install & run via `uv` |
| **Headless / CI mode** | Runs without a display; prints reasoning to stdout |

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager

```bash
pip install uv      # if uv is not yet installed
```

### Install & run (graphical)

```bash
git clone https://github.com/hubbertj/ai-801-wumpus-world
cd ai-801-wumpus-world
uv sync
uv run wumpus-world
```

A difficulty-selection menu appears. Use **↑ / ↓** or **1 / 2 / 3** to choose, then **Enter**.

### Running with 2D graphics

The simulation uses **Pygame** for a 2D graphical window. To run with the full 2D display:

1. **Use a machine with a display** (local desktop or a session with `DISPLAY` / `WAYLAND_DISPLAY` set). On headless servers or CI, the app will fall back to terminal-only mode unless you force graphics (see below).

2. **Install and run** (no extra flags needed for graphics):
   ```bash
   uv sync
   uv run wumpus-world
   ```

3. **What you get:** A Pygame window with:
   - **Actual World** (left): fog-of-war grid; pits, Wumpus, and gold drawn as 2D graphics; agent sprite with direction arrow.
   - **Agent Knowledge Base** (centre): same 2D art for confirmed pits/Wumpus; colour-coded beliefs.
   - **Sidebar** (right): KB stats, legend, and live AI reasoning log.

4. **Force graphical mode:** To insist on the Pygame window (e.g. if auto-detect is wrong), use:
   ```bash
   uv run wumpus-world --graphical
   # or
   uv run wumpus-world -g
   ```
   If no window appears, ensure `DISPLAY` (X11) or `WAYLAND_DISPLAY` (Wayland) is set and that `WUMPUS_HEADLESS` is not set.

### Headless / terminal mode

```bash
# Easy (default)
WUMPUS_HEADLESS=1 uv run wumpus-world

# Specific difficulty
WUMPUS_HEADLESS=1 uv run wumpus-world --medium
WUMPUS_HEADLESS=1 uv run wumpus-world --hard
```

---

## Controls (Pygame window)

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `→` | Single-step (when paused) |
| `↑` / `↓` | Faster / slower auto-step |
| `Q` / `Esc` | Quit |

---

## Difficulty Levels

| Level | Grid | Pits | Wumpus |
|-------|------|------|--------|
| **Easy** | 4 × 4 | 2 | No |
| **Medium** | 6 × 6 | 4 | Yes |
| **Hard** | 8 × 8 | 8 | Yes |

All dangerous elements (pits and Wumpus) are placed **outside** the cells adjacent to the start position `(0,0)`, so the agent can always take at least one safe step.

---

## Display Layout

The 2D graphics window shows two grids plus a sidebar. Cells use drawn graphics for the agent (explorer + direction arrow), pits (holes), Wumpus (monster face), and gold (coin); stench and breeze are shown as small icons.

```
┌─────────────────────┬──────────────────────┬───────────────────────────┐
│  ACTUAL WORLD       │  AGENT KNOWLEDGE     │  AI AGENT STATUS          │
│  (fog of war)       │  BASE                │  ─ KB statistics          │
│                     │                      │  ─ Confirmed dangers      │
│  Only visited cells │  Color-coded belief  │  ─ Map legend             │
│  are revealed.      │  state of every cell │  ─ AI REASONING LOG       │
│                     │                      │    (scrolling, live)      │
└─────────────────────┴──────────────────────┴───────────────────────────┘
```

### Knowledge Base colour codes

| Colour | Meaning |
|--------|---------|
| Dark green | Visited (safe, explored) |
| Bright green | Safe but not yet explored |
| Orange-brown | Cell may contain a pit |
| Dark rose | Cell may contain the Wumpus |
| Dark red | Both pit and Wumpus possible |
| Blue | **Confirmed** pit |
| Deep red | **Confirmed** Wumpus |
| Near-black | Completely unknown |

---

## How the AI Works

The agent is a **knowledge-based agent** that maintains a propositional logical knowledge base (KB) and updates it at every step using the following rules:

### Perception → Inference

| Perception | Inference |
|-----------|-----------|
| No **Breeze** at cell X | None of X's neighbours contain a pit |
| **Breeze** at X | ≥1 unvisited neighbour may contain a pit |
| No **Stench** at X | None of X's neighbours contain the Wumpus |
| **Stench** at X | ≥1 unvisited neighbour may contain the Wumpus |
| **Glitter** at X | Gold is here — GRAB immediately |
| **Scream** heard | Wumpus is dead; all stench-danger eliminated |
| **Bump** felt | Walked into a wall |

### Constraint propagation

After each update, the agent runs inference passes over all visited cells:

- **Pit confirmation**: if a breezy cell has exactly one unresolved pit candidate (and no confirmed pit already explains the breeze) → that cell **must** be a pit.
- **Wumpus confirmation**: same logic for stench / Wumpus candidates.
- **Cross-constraint**: a cell can never be simultaneously confirmed pit *and* confirmed Wumpus (the game places them exclusively).

### Goal-directed planning

```
Priority 1: Grab gold (if Glitter perceived)
Priority 2: Return to (0,0) and Climb (if carrying gold)
Priority 3: Shoot confirmed Wumpus (if arrow available and path exists)
Priority 4: Navigate to nearest safe unvisited cell (BFS)
Priority 5: Take calculated risk on least-dangerous frontier cell
Priority 6: Return home and Climb (no more options)
```

Path planning uses **BFS on the grid** to produce a sequence of `TURN_LEFT / TURN_RIGHT / MOVE_FORWARD` actions, correctly accounting for the agent's current facing direction.

---

## Project Structure

```
wumpus-world/
├── pyproject.toml               # UV project config & entry point
├── uv.lock                      # Reproducible dependency lock
├── src/
│   └── wumpus_world/
│       ├── __init__.py
│       ├── environment.py       # Grid, perceptions, game rules
│       ├── knowledge_base.py    # Logical KB & constraint inference
│       ├── agent.py             # AI agent with planning & reasoning
│       ├── renderer.py          # Pygame 2D display
│       └── main.py              # Entry point, difficulty menu
└── tests/
    └── test_wumpus_world.py     # 49 pytest unit tests
```

---

## Running Tests

```bash
uv run pytest
```

Expected output: **49 passed**.

---

## Scoring

| Event | Score |
|-------|-------|
| Each action taken | −1 |
| Arrow fired | −10 |
| Fell into pit / eaten | −1 000 |
| Grabbed the gold | +1 000 |
| Climbed out alive with gold | +500 (bonus) |
