"""AI Agent for Wumpus World.

Uses a logical knowledge base and goal-directed planning to:
  1. Safely explore the cave
  2. Grab the gold
  3. Return to the start and climb out

The agent explains every decision it makes.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from wumpus_world.environment import Action, Direction, Perception, WumpusWorld
from wumpus_world.knowledge_base import KnowledgeBase


class Agent:
    """Knowledge-based AI agent for Wumpus World."""

    def __init__(self, world: WumpusWorld):
        self.world = world
        self.kb = KnowledgeBase(world.size)

        # Agent's own tracked state (mirrors world for planning)
        self.pos: tuple[int, int] = (0, 0)
        self.direction: Direction = Direction.EAST
        self.has_gold: bool = False

        # Pending action sequence
        self.plan: list[Action] = []

        # Counters / logs
        self.step_count: int = 0
        self.decision_log: list[str] = []

        # Shoot-planning target
        self._shoot_goal: Optional[tuple[int, int]] = None

        # Inform KB of initial position (no perceptions yet before first move)
        initial_percepts = world.get_percepts()
        self.kb.update(self.pos, initial_percepts)

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def _decide(self, msg: str) -> None:
        """Record a high-level decision with step number."""
        entry = f"Step {self.step_count}: {msg}"
        self.decision_log.append(entry)
        self.kb.reasoning_log.append(f"[DECISION] {msg}")

    def choose_action(self) -> Action:
        """Select the next action based on the current knowledge base."""
        self.step_count += 1
        percepts = self.world.get_percepts()

        # ── 1. Grab gold if we see glitter ───────────────────────────
        if Perception.GLITTER in percepts and not self.has_gold:
            self._decide("Glitter detected at my position — grabbing the gold!")
            return Action.GRAB

        # ── 2. Climb out if we have gold and are at start ─────────────
        if self.has_gold:
            if self.pos == (0, 0):
                self._decide("I have the gold AND I'm at the entrance. Climbing out!")
                return Action.CLIMB
            if not self.plan:
                self._decide(
                    f"I have the gold! Planning route home from {self.pos} to (0,0)."
                )
                path = self._find_path(self.pos, (0, 0), safe_only=False)
                if path:
                    self.plan = path
                    self._decide(f"Route home: {[a.name for a in path]}")
                else:
                    self._decide("No path home found — trying to climb anyway.")
                    return Action.CLIMB
            if self.plan:
                return self.plan.pop(0)

        # ── 3. Execute pending plan ───────────────────────────────────
        if self.plan:
            return self.plan.pop(0)

        # ── 4. Shoot confirmed Wumpus if possible ────────────────────
        if (
            self.kb.confirmed_wumpus
            and not self.kb.wumpus_dead
            and self.kb.has_arrow
        ):
            shoot_plan = self._plan_shoot(self.kb.confirmed_wumpus)
            if shoot_plan:
                self._decide(
                    f"Confirmed Wumpus at {self.kb.confirmed_wumpus}. "
                    f"Planning to shoot: {[a.name for a in shoot_plan]}"
                )
                self.plan = shoot_plan[1:]
                return shoot_plan[0]

        # ── 5. Explore safe unvisited cells ──────────────────────────
        safe_frontier = self.kb.get_safe_unvisited()
        if safe_frontier:
            target = self._nearest(safe_frontier)
            path = self._find_path(self.pos, target, safe_only=True)
            if path:
                self._decide(
                    f"Safe unvisited cells available. Heading to {target} "
                    f"({len(safe_frontier)} safe options). "
                    f"Path: {[a.name for a in path]}"
                )
                self.plan = path[1:]
                return path[0]

        # ── 6. Take a calculated risk on the least dangerous frontier ─
        frontier = self._get_risky_frontier()
        if frontier:
            def danger_score(cell: tuple[int, int]) -> int:
                score = 0
                if cell in self.kb.pit_possible:
                    score += 2
                if cell in self.kb.wumpus_possible:
                    score += 3
                return score

            target = min(
                frontier,
                key=lambda c: (danger_score(c), abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1])),
            )
            d = danger_score(target)
            self._decide(
                f"No confirmed safe cells. Taking calculated risk on {target} "
                f"(danger={d}). "
                + ("May contain pit!" if target in self.kb.pit_possible else "")
                + (" May contain Wumpus!" if target in self.kb.wumpus_possible else "")
            )
            path = self._find_path(self.pos, target, safe_only=False)
            if path:
                self.plan = path[1:]
                return path[0]

        # ── 7. Give up — head home ────────────────────────────────────
        if self.pos == (0, 0):
            self._decide("No moves available. Climbing out without gold.")
            return Action.CLIMB
        self._decide("Explored everything reachable. Returning to entrance.")
        path = self._find_path(self.pos, (0, 0), safe_only=False)
        if path:
            self.plan = path[1:]
            return path[0]
        return Action.CLIMB

    # ------------------------------------------------------------------
    # State update after action
    # ------------------------------------------------------------------

    def update(self, result: dict) -> None:
        """Update the agent's state and knowledge base after an action."""
        action = result["action"]
        percepts: frozenset[Perception] = frozenset(result.get("percepts", set()))

        if action == Action.MOVE_FORWARD:
            if Perception.BUMP not in percepts:
                # Sync position/direction from world
                self.pos = self.world.agent_pos
                self.direction = self.world.agent_dir
                if not result.get("died"):
                    self.kb.update(self.pos, percepts)
                    if Perception.SCREAM in percepts:
                        self.kb.mark_wumpus_dead()

        elif action == Action.TURN_LEFT:
            self.direction = self.world.agent_dir

        elif action == Action.TURN_RIGHT:
            self.direction = self.world.agent_dir

        elif action == Action.GRAB:
            if self.world.agent_has_gold:
                self.has_gold = True
                self.kb.mark_gold_grabbed()
                self.plan = []  # Rebuild plan to go home

        elif action == Action.SHOOT:
            self.kb.mark_arrow_used()
            if Perception.SCREAM in percepts:
                self.kb.mark_wumpus_dead()

        # Print reasoning to console as well
        if result.get("message"):
            print(f"  World → {result['message']}")

    # ------------------------------------------------------------------
    # Path-finding (BFS on grid → action sequence)
    # ------------------------------------------------------------------

    def _find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        safe_only: bool = True,
    ) -> Optional[list[Action]]:
        """BFS from *start* to *goal*; returns a list of Actions or None."""
        if start == goal:
            return []

        def can_traverse(pos: tuple[int, int]) -> bool:
            if pos == goal:
                return True
            if pos in self.kb.confirmed_pits:
                return False
            if pos == self.kb.confirmed_wumpus and not self.kb.wumpus_dead:
                return False
            if safe_only and pos not in self.kb.safe_cells:
                return False
            return True

        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque(
            [(start, [])]
        )
        seen: set[tuple[int, int]] = {start}

        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return self._positions_to_actions(start, path)

            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                npos = (r + dr, c + dc)
                nr, nc = npos
                if (
                    0 <= nr < self.world.size
                    and 0 <= nc < self.world.size
                    and npos not in seen
                    and can_traverse(npos)
                ):
                    seen.add(npos)
                    queue.append((npos, path + [npos]))
        return None

    def _positions_to_actions(
        self,
        start: tuple[int, int],
        positions: list[tuple[int, int]],
    ) -> list[Action]:
        """Convert a list of grid positions into a sequence of Actions."""
        actions: list[Action] = []
        current_dir = self.direction
        current_pos = start

        for next_pos in positions:
            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]

            if dr == -1:
                target_dir = Direction.NORTH
            elif dr == 1:
                target_dir = Direction.SOUTH
            elif dc == 1:
                target_dir = Direction.EAST
            else:
                target_dir = Direction.WEST

            turns = self._turns_needed(current_dir, target_dir)
            actions.extend(turns)
            current_dir = target_dir
            actions.append(Action.MOVE_FORWARD)
            current_pos = next_pos

        return actions

    @staticmethod
    def _turns_needed(current: Direction, target: Direction) -> list[Action]:
        """Minimum turns to go from *current* to *target* direction."""
        diff = (target.value - current.value) % 4
        if diff == 0:
            return []
        if diff == 1:
            return [Action.TURN_RIGHT]
        if diff == 2:
            return [Action.TURN_RIGHT, Action.TURN_RIGHT]
        # diff == 3
        return [Action.TURN_LEFT]

    # ------------------------------------------------------------------
    # Shooting
    # ------------------------------------------------------------------

    def _plan_shoot(
        self, wumpus_pos: tuple[int, int]
    ) -> Optional[list[Action]]:
        """Return action sequence to navigate adjacent to wumpus and shoot it."""
        wr, wc = wumpus_pos

        # Adjacent positions from which a shot would hit the wumpus
        candidates: list[tuple[tuple[int, int], Direction]] = [
            ((wr + 1, wc), Direction.NORTH),   # south of wumpus → face north
            ((wr - 1, wc), Direction.SOUTH),   # north of wumpus → face south
            ((wr, wc + 1), Direction.WEST),    # east of wumpus  → face west
            ((wr, wc - 1), Direction.EAST),    # west of wumpus  → face east
        ]

        for shoot_pos, face_dir in candidates:
            sr, sc = shoot_pos
            if not (0 <= sr < self.world.size and 0 <= sc < self.world.size):
                continue
            if shoot_pos not in self.kb.safe_cells:
                continue
            travel = self._find_path(self.pos, shoot_pos, safe_only=True)
            if travel is None:
                continue

            # After travel we will be at shoot_pos.  Simulate the final
            # facing direction so we can add the correct turns to aim.
            sim_dir = self._simulate_final_direction(self.direction, travel)
            aim_turns = self._turns_needed(sim_dir, face_dir)
            return travel + aim_turns + [Action.SHOOT]

        return None

    def _simulate_final_direction(
        self, start_dir: Direction, actions: list[Action]
    ) -> Direction:
        """Simulate direction changes from a sequence of actions."""
        d = start_dir
        for a in actions:
            if a == Action.TURN_LEFT:
                d = Direction((d.value - 1) % 4)
            elif a == Action.TURN_RIGHT:
                d = Direction((d.value + 1) % 4)
        return d

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest(self, cells: set[tuple[int, int]]) -> tuple[int, int]:
        """Return the cell with the smallest Manhattan distance to agent."""
        return min(
            cells,
            key=lambda c: abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]),
        )

    def _get_risky_frontier(self) -> list[tuple[int, int]]:
        """Return unvisited cells adjacent to visited cells (excluding confirmed danger)."""
        risky: list[tuple[int, int]] = []
        for (r, c) in self.kb.visited:
            for p in self.kb._adjacent(r, c):
                if (
                    p not in self.kb.visited
                    and p not in self.kb.safe_cells
                    and p not in self.kb.confirmed_pits
                    and p != self.kb.confirmed_wumpus
                    and p not in risky
                ):
                    risky.append(p)
        return risky
