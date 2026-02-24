"""Wumpus World environment: grid, perceptions, game logic."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Action(Enum):
    MOVE_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    GRAB = auto()
    SHOOT = auto()
    CLIMB = auto()


class Perception(Enum):
    STENCH = "stench"
    BREEZE = "breeze"
    GLITTER = "glitter"
    BUMP = "bump"
    SCREAM = "scream"


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


DIFFICULTY_CONFIGS = {
    DifficultyLevel.EASY: {
        "size": 4,
        "num_pits": 2,
        "has_wumpus": False,
        "description": "4×4 grid · 2 pits · No Wumpus",
    },
    DifficultyLevel.MEDIUM: {
        "size": 6,
        "num_pits": 4,
        "has_wumpus": True,
        "description": "6×6 grid · 4 pits · 1 Wumpus",
    },
    DifficultyLevel.HARD: {
        "size": 8,
        "num_pits": 8,
        "has_wumpus": True,
        "description": "8×8 grid · 8 pits · 1 Wumpus",
    },
}


@dataclass
class Cell:
    has_wumpus: bool = False
    has_pit: bool = False
    has_gold: bool = False
    has_stench: bool = False
    has_breeze: bool = False


class WumpusWorld:
    """Represents the Wumpus World environment."""

    def __init__(self, difficulty: DifficultyLevel, seed: Optional[int] = None):
        config = DIFFICULTY_CONFIGS[difficulty]
        self.size: int = config["size"]
        self.num_pits: int = config["num_pits"]
        self.difficulty = difficulty

        if seed is not None:
            random.seed(seed)

        self.grid: list[list[Cell]] = [
            [Cell() for _ in range(self.size)] for _ in range(self.size)
        ]
        self.wumpus_pos: Optional[tuple[int, int]] = None
        self.gold_pos: Optional[tuple[int, int]] = None
        self.has_wumpus: bool = config["has_wumpus"]

        # Agent state
        self.agent_pos: tuple[int, int] = (0, 0)
        self.agent_dir: Direction = Direction.EAST
        self.agent_has_arrow: bool = True
        self.agent_has_gold: bool = False
        self.wumpus_alive: bool = self.has_wumpus

        # Game state
        self.game_over: bool = False
        self.won: bool = False
        self.score: int = 0

        self._generate_world()

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------

    def _generate_world(self) -> None:
        """Randomly place Wumpus, gold, and pits.

        Cells adjacent to the start (0,0) are kept safe so the agent can
        always take at least one step without immediately dying.
        """
        # Cells adjacent to start are kept safe for placement of dangers
        start_adj = set(self._adjacent(0, 0))

        # Candidate cells for dangerous elements (pits / wumpus):
        # must not be start or directly adjacent to start.
        danger_candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) != (0, 0) and (r, c) not in start_adj
        ]

        # Gold can be anywhere except start
        all_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) != (0, 0)
        ]
        random.shuffle(all_cells)
        random.shuffle(danger_candidates)

        # Place gold (can be anywhere except start)
        self.gold_pos = all_cells.pop()
        self.grid[self.gold_pos[0]][self.gold_pos[1]].has_gold = True

        # Place Wumpus in danger-only zone (not adjacent to start)
        if self.has_wumpus and danger_candidates:
            self.wumpus_pos = danger_candidates.pop()
            self.grid[self.wumpus_pos[0]][self.wumpus_pos[1]].has_wumpus = True

        # Place pits in danger-only zone, avoiding gold/wumpus cells
        used = {self.gold_pos}
        if self.wumpus_pos:
            used.add(self.wumpus_pos)
        pit_slots = [p for p in danger_candidates if p not in used]
        for i in range(min(self.num_pits, len(pit_slots))):
            self.grid[pit_slots[i][0]][pit_slots[i][1]].has_pit = True

        self._update_perceptions()

    def _update_perceptions(self) -> None:
        """Recompute stench/breeze perceptions across the entire grid."""
        for r in range(self.size):
            for c in range(self.size):
                self.grid[r][c].has_stench = False
                self.grid[r][c].has_breeze = False

        if self.wumpus_pos and self.wumpus_alive:
            wr, wc = self.wumpus_pos
            for nr, nc in self._adjacent(wr, wc):
                self.grid[nr][nc].has_stench = True

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c].has_pit:
                    for nr, nc in self._adjacent(r, c):
                        self.grid[nr][nc].has_breeze = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjacent(self, r: int, c: int) -> list[tuple[int, int]]:
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                result.append((nr, nc))
        return result

    def _direction_delta(self, direction: Direction) -> tuple[int, int]:
        return {
            Direction.NORTH: (-1, 0),
            Direction.EAST: (0, 1),
            Direction.SOUTH: (1, 0),
            Direction.WEST: (0, -1),
        }[direction]

    # ------------------------------------------------------------------
    # Perceptions
    # ------------------------------------------------------------------

    def get_percepts(self) -> frozenset[Perception]:
        """Return the set of current perceptions at the agent's position."""
        r, c = self.agent_pos
        cell = self.grid[r][c]
        percepts: set[Perception] = set()
        if cell.has_stench:
            percepts.add(Perception.STENCH)
        if cell.has_breeze:
            percepts.add(Perception.BREEZE)
        if cell.has_gold and not self.agent_has_gold:
            percepts.add(Perception.GLITTER)
        return frozenset(percepts)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> dict:
        """Apply an action and return a result dict with feedback."""
        result: dict = {
            "action": action,
            "percepts": set(),
            "message": "",
            "died": False,
            "won": False,
        }
        self.score -= 1  # Step cost

        if action == Action.MOVE_FORWARD:
            dr, dc = self._direction_delta(self.agent_dir)
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

            if 0 <= nr < self.size and 0 <= nc < self.size:
                self.agent_pos = (nr, nc)
                result["message"] = f"Moved forward to ({nr}, {nc})"
                if self.grid[nr][nc].has_pit:
                    self.game_over = True
                    result["died"] = True
                    result["message"] = f"💀 Fell into a pit at ({nr}, {nc})! Game over."
                    self.score -= 1000
                elif self.grid[nr][nc].has_wumpus and self.wumpus_alive:
                    self.game_over = True
                    result["died"] = True
                    result["message"] = f"💀 Eaten by the Wumpus at ({nr}, {nc})! Game over."
                    self.score -= 1000
            else:
                result["percepts"].add(Perception.BUMP)
                result["message"] = "Bumped into a wall!"

        elif action == Action.TURN_LEFT:
            self.agent_dir = Direction((self.agent_dir.value - 1) % 4)
            result["message"] = f"Turned left → now facing {self.agent_dir.name}"

        elif action == Action.TURN_RIGHT:
            self.agent_dir = Direction((self.agent_dir.value + 1) % 4)
            result["message"] = f"Turned right → now facing {self.agent_dir.name}"

        elif action == Action.GRAB:
            r, c = self.agent_pos
            if self.grid[r][c].has_gold and not self.agent_has_gold:
                self.agent_has_gold = True
                self.grid[r][c].has_gold = False
                self.score += 1000
                result["message"] = "✨ Grabbed the gold! Head back to (0,0)!"
            else:
                result["message"] = "Nothing to grab here."

        elif action == Action.SHOOT:
            if self.agent_has_arrow:
                self.agent_has_arrow = False
                self.score -= 10
                result["message"] = "🏹 Arrow fired!"

                if self.wumpus_pos and self.wumpus_alive:
                    dr, dc = self._direction_delta(self.agent_dir)
                    r, c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
                    hit = False
                    while 0 <= r < self.size and 0 <= c < self.size:
                        if (r, c) == self.wumpus_pos:
                            self.wumpus_alive = False
                            self.grid[self.wumpus_pos[0]][self.wumpus_pos[1]].has_wumpus = False
                            self._update_perceptions()
                            result["percepts"].add(Perception.SCREAM)
                            result["message"] = "🏹 Arrow hit the Wumpus! A blood-curdling SCREAM echoes..."
                            hit = True
                            break
                        r += dr
                        c += dc
                    if not hit:
                        result["message"] = "🏹 Arrow missed. Wasted."
            else:
                result["message"] = "No arrow remaining."

        elif action == Action.CLIMB:
            if self.agent_pos == (0, 0):
                self.game_over = True
                if self.agent_has_gold:
                    self.won = True
                    result["won"] = True
                    result["message"] = "🏆 Climbed out WITH the gold! YOU WIN!"
                    self.score += 500  # Bonus for escaping alive
                else:
                    result["message"] = "Climbed out without the gold. Game over."
            else:
                result["message"] = "Can only climb out from starting position (0, 0)."

        # Append current cell percepts to result
        result["percepts"].update(self.get_percepts())
        return result
