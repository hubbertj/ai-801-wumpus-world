"""Knowledge Base for the Wumpus World agent.

Maintains logical inferences about the state of the world
based on perceptions gathered during exploration.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from wumpus_world.environment import Perception


class CellStatus(Enum):
    UNKNOWN = "unknown"
    SAFE = "safe"
    VISITED = "visited"
    POSSIBLY_PIT = "possibly_pit"
    POSSIBLY_WUMPUS = "possibly_wumpus"
    DANGEROUS = "dangerous"
    CONFIRMED_PIT = "confirmed_pit"
    CONFIRMED_WUMPUS = "confirmed_wumpus"


class KnowledgeBase:
    """Logical knowledge base used by the agent to reason about the world."""

    def __init__(self, size: int):
        self.size = size

        # What the agent has directly observed
        self.visited: dict[tuple[int, int], frozenset[Perception]] = {}

        # Derived knowledge sets
        self.safe_cells: set[tuple[int, int]] = set()
        self.pit_possible: set[tuple[int, int]] = set()
        self.wumpus_possible: set[tuple[int, int]] = set()
        self.confirmed_pits: set[tuple[int, int]] = set()
        self.confirmed_wumpus: Optional[tuple[int, int]] = None

        # Agent item state mirrored for KB reasoning
        self.wumpus_dead: bool = False
        self.has_arrow: bool = True
        self.has_gold: bool = False

        # Human-readable reasoning log
        self.reasoning_log: list[str] = []

        # Start position is known safe
        self.safe_cells.add((0, 0))

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

    def _log(self, msg: str) -> None:
        self.reasoning_log.append(msg)

    def _recompute_safe(self, cell: tuple[int, int]) -> None:
        """Mark a cell as safe if it is definitively free of pits and wumpus."""
        if cell in self.visited:
            self.safe_cells.add(cell)
            return
        if cell in self.confirmed_pits:
            self.safe_cells.discard(cell)
            return
        if cell == self.confirmed_wumpus and not self.wumpus_dead:
            self.safe_cells.discard(cell)
            return
        if cell not in self.pit_possible and cell not in self.wumpus_possible:
            self.safe_cells.add(cell)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, pos: tuple[int, int], percepts: frozenset[Perception]) -> None:
        """Incorporate new perceptions at *pos* into the knowledge base."""
        r, c = pos

        self.visited[pos] = frozenset(percepts)
        self.safe_cells.add(pos)
        self.pit_possible.discard(pos)
        self.wumpus_possible.discard(pos)

        adj = self._adjacent(r, c)
        unvisited_adj = [p for p in adj if p not in self.visited]

        # ── Scream (wumpus died) ──────────────────────────────────────
        if Perception.SCREAM in percepts:
            self.wumpus_dead = True
            self.confirmed_wumpus = None
            old_count = len(self.wumpus_possible)
            self.wumpus_possible.clear()
            self._log(
                f"[SCREAM] Wumpus killed! Cleared {old_count} wumpus-possible cells."
            )

        # ── Breeze inference ─────────────────────────────────────────
        if Perception.BREEZE not in percepts:
            removed = []
            for p in adj:
                if p in self.pit_possible:
                    self.pit_possible.discard(p)
                    removed.append(p)
            # Do NOT recompute safety here; wait until stench is also processed
            # so wumpus_possible is fully updated first.
            self._log(
                f"[SAFE] No breeze at {pos} → no adjacent pits."
                + (f" Cleared pit-possible: {removed}" if removed else "")
            )
        else:
            added = []
            for p in unvisited_adj:
                if p not in self.safe_cells and p not in self.confirmed_pits:
                    self.pit_possible.add(p)
                    added.append(p)
            self._log(
                f"[BREEZE] Breeze at {pos} → adjacent cells may hide pits: {added}"
            )

        # ── Stench inference ─────────────────────────────────────────
        if Perception.STENCH not in percepts:
            removed = []
            for p in adj:
                if p in self.wumpus_possible:
                    self.wumpus_possible.discard(p)
                    removed.append(p)
            self._log(
                f"[SAFE] No stench at {pos} → no adjacent Wumpus."
                + (f" Cleared wumpus-possible: {removed}" if removed else "")
            )
        else:
            if not self.wumpus_dead:
                added = []
                for p in unvisited_adj:
                    if p not in self.safe_cells and p not in self.confirmed_wumpus_set():
                        self.wumpus_possible.add(p)
                        added.append(p)
                self._log(
                    f"[STENCH] Stench at {pos} → Wumpus may lurk in: {added}"
                )
            else:
                self._log(f"[STENCH] Stench at {pos}, but Wumpus already dead. Ignoring.")

        # ── Recompute safety for all adjacent cells now that both pit_possible
        #    and wumpus_possible have been fully updated for this percept set ──
        for p in adj:
            self._recompute_safe(p)

        # ── Glitter ──────────────────────────────────────────────────
        if Perception.GLITTER in percepts:
            self._log(f"[GLITTER] ✨ Gold detected at {pos}! I should GRAB it now!")

        # ── Run deeper inference ──────────────────────────────────────
        self._infer()

    def confirmed_wumpus_set(self) -> set[tuple[int, int]]:
        return {self.confirmed_wumpus} if self.confirmed_wumpus else set()

    # ------------------------------------------------------------------
    # Logical inference
    # ------------------------------------------------------------------

    def _infer(self) -> None:
        """Apply constraint propagation to confirm pit/wumpus locations.

        A cell cannot simultaneously be a confirmed pit AND a confirmed wumpus,
        so we guard against cross-confirmation of the same location.
        """
        for (r, c), percepts in self.visited.items():
            adj = self._adjacent(r, c)

            # ── Pit inference ─────────────────────────────────────
            if Perception.BREEZE in percepts:
                # If a confirmed pit already explains this breeze, skip —
                # we cannot eliminate other candidates based on this cell.
                already_explained = any(p in self.confirmed_pits for p in adj)
                if not already_explained:
                    candidates = [p for p in adj if p in self.pit_possible]
                    if len(candidates) == 1:
                        pit = candidates[0]
                        # Never confirm a pit at the same cell as a confirmed wumpus
                        if pit not in self.confirmed_pits and pit != self.confirmed_wumpus:
                            self.confirmed_pits.add(pit)
                            self.pit_possible.discard(pit)
                            self.safe_cells.discard(pit)
                            self._log(
                                f"[INFER] Only unresolved pit candidate adjacent to {(r,c)}"
                                f" → CONFIRMED PIT at {pit}!"
                            )

            # ── Wumpus inference ──────────────────────────────────
            if Perception.STENCH in percepts and not self.wumpus_dead:
                # If confirmed wumpus already explains this stench, skip.
                already_explained = (
                    self.confirmed_wumpus is not None and self.confirmed_wumpus in adj
                )
                if not already_explained:
                    candidates = [p for p in adj if p in self.wumpus_possible]
                    if len(candidates) == 1:
                        wpos = candidates[0]
                        # Never confirm wumpus at the same cell as a confirmed pit
                        if self.confirmed_wumpus != wpos and wpos not in self.confirmed_pits:
                            self.confirmed_wumpus = wpos
                            self.safe_cells.discard(wpos)
                            self._log(
                                f"[INFER] Only one wumpus candidate adjacent to {(r,c)}"
                                f" → CONFIRMED WUMPUS at {wpos}!"
                            )

        # ── Cross-constraint: if confirmed wumpus ≠ None,
        #    remove it from safe / pit-possible
        if self.confirmed_wumpus:
            self.safe_cells.discard(self.confirmed_wumpus)

        for pit in self.confirmed_pits:
            self.safe_cells.discard(pit)
            self.pit_possible.discard(pit)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_safe_unvisited(self) -> set[tuple[int, int]]:
        """Return cells that are confirmed safe but not yet explored."""
        return self.safe_cells - set(self.visited.keys())

    def get_cell_status(self, pos: tuple[int, int]) -> CellStatus:
        if pos in self.confirmed_pits:
            return CellStatus.CONFIRMED_PIT
        if self.confirmed_wumpus == pos and not self.wumpus_dead:
            return CellStatus.CONFIRMED_WUMPUS
        if pos in self.visited:
            return CellStatus.VISITED
        if pos in self.safe_cells:
            return CellStatus.SAFE
        in_pit = pos in self.pit_possible
        in_wumpus = pos in self.wumpus_possible
        if in_pit and in_wumpus:
            return CellStatus.DANGEROUS
        if in_pit:
            return CellStatus.POSSIBLY_PIT
        if in_wumpus:
            return CellStatus.POSSIBLY_WUMPUS
        return CellStatus.UNKNOWN

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------

    def mark_gold_grabbed(self) -> None:
        self.has_gold = True
        self._log(
            "[ACTION] Gold is mine! Objective: navigate back to (0,0) and CLIMB OUT."
        )

    def mark_arrow_used(self) -> None:
        self.has_arrow = False
        self._log("[ACTION] Arrow used. Only one shot per game — spent.")

    def mark_wumpus_dead(self) -> None:
        self.wumpus_dead = True
        self.confirmed_wumpus = None
        self.wumpus_possible.clear()
        self._log("[SCREAM] Wumpus is dead! All stench-related danger eliminated.")
