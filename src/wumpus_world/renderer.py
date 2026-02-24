"""Pygame renderer for the Wumpus World simulation.

Layout
------
┌─────────────────┬─────────────────┬─────────────────────────────┐
│  ACTUAL WORLD   │ AGENT KNOWLEDGE │  STATUS / AI REASONING LOG  │
│  (fog of war)   │  (known map)    │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
"""

from __future__ import annotations

import os
from typing import Optional

# Allow importing without a display (for unit tests / CI)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from wumpus_world.agent import Agent
from wumpus_world.environment import Direction, Perception, WumpusWorld
from wumpus_world.knowledge_base import CellStatus, KnowledgeBase

# ── Palette ───────────────────────────────────────────────────────────────────
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_BG = (18, 20, 30)
PANEL_BG = (22, 26, 40)
BORDER = (55, 75, 120)
GRAY = (110, 115, 130)
LIGHT_GRAY = (195, 200, 210)

GREEN = (45, 195, 60)
LIGHT_GREEN = (140, 225, 145)
DARK_GREEN = (25, 80, 30)

RED = (215, 55, 55)
DARK_RED = (140, 15, 15)

ORANGE = (230, 148, 40)
BLUE = (55, 105, 225)
DARK_BLUE = (15, 40, 145)
YELLOW = (252, 220, 0)
PURPLE = (148, 50, 200)
TEAL = (0, 180, 178)
PINK = (255, 105, 180)
FOG = (12, 14, 22)

# Cell background colours for the world grid
CELL_VISITED = (45, 52, 65)
CELL_FOG = FOG
CELL_AGENT = (28, 75, 195)
CELL_PIT = (8, 15, 90)
CELL_WUMPUS = (95, 8, 8)

# Cell background colours for the knowledge map
KB_VISITED = (30, 60, 30)
KB_SAFE = (55, 135, 60)
KB_UNKNOWN = (35, 38, 50)
KB_PIT_POSSIBLE = (90, 48, 8)
KB_WUMPUS_POSSIBLE = (90, 8, 45)
KB_DANGEROUS = (110, 15, 15)
KB_CONFIRMED_PIT = (10, 10, 115)
KB_CONFIRMED_WUMPUS = (120, 8, 8)
KB_AGENT = (28, 75, 195)


class WumpusRenderer:
    """Main Pygame-based renderer for the Wumpus World simulation."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        world: WumpusWorld,
        agent: Agent,
        headless: bool = False,
    ):
        self.world = world
        self.agent = agent
        self.headless = headless

        if not headless:
            os.environ.pop("SDL_VIDEODRIVER", None)

        pygame.init()
        pygame.display.init()

        # Grid cell size (scales with world size)
        self.cell_size = min(88, 510 // world.size)
        self.grid_px = self.cell_size * world.size

        # Layout constants
        self.HEADER_H = 64
        self.PADDING = 10
        self.SIDEBAR_W = 480

        total_w = self.grid_px * 2 + self.SIDEBAR_W + self.PADDING * 4
        total_h = max(
            self.grid_px + self.HEADER_H + self.PADDING * 3 + 24,
            720,
        )

        flags = pygame.NOFRAME if headless else 0
        self.screen = pygame.display.set_mode((total_w, total_h), flags)
        pygame.display.set_caption("🌍 Wumpus World AI Simulation")

        # Typography
        self.f_title = pygame.font.SysFont("monospace", 18, bold=True)
        self.f_body = pygame.font.SysFont("monospace", 14)
        self.f_small = pygame.font.SysFont("monospace", 11)
        self.f_label = pygame.font.SysFont("monospace", 10)

        # Control state
        self.running: bool = True
        self.paused: bool = False
        self.step_delay: int = 600   # ms between auto-steps
        self.last_step_ms: int = 0
        self.clock = pygame.time.Clock()

        # Layout anchors
        self._world_x = self.PADDING
        self._world_y = self.HEADER_H + self.PADDING + 20
        self._kb_x = self.grid_px + self.PADDING * 2
        self._kb_y = self._world_y
        self._side_x = self.grid_px * 2 + self.PADDING * 3
        self._side_y = self.PADDING

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self) -> Optional[str]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif k == pygame.K_SPACE:
                    self.paused = not self.paused
                    tag = "PAUSED" if self.paused else "RESUMED"
                    print(f"  [{tag}]")
                elif k == pygame.K_RIGHT and self.paused:
                    return "STEP"
                elif k == pygame.K_UP:
                    self.step_delay = max(100, self.step_delay - 100)
                    print(f"  Speed: {self.step_delay} ms/step")
                elif k == pygame.K_DOWN:
                    self.step_delay = min(3000, self.step_delay + 100)
                    print(f"  Speed: {self.step_delay} ms/step")
        return None

    # ------------------------------------------------------------------
    # Master draw
    # ------------------------------------------------------------------

    def draw(self) -> None:
        self.screen.fill(DARK_BG)
        self._draw_header()
        self._draw_world_grid()
        self._draw_kb_grid()
        self._draw_sidebar()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Header bar
    # ------------------------------------------------------------------

    def _draw_header(self) -> None:
        pygame.draw.rect(
            self.screen, (18, 30, 68), (0, 0, self.screen.get_width(), self.HEADER_H)
        )
        pygame.draw.line(
            self.screen, BORDER, (0, self.HEADER_H - 1), (self.screen.get_width(), self.HEADER_H - 1)
        )

        title = self.f_title.render(
            f"WUMPUS WORLD  ·  {self.world.difficulty.value.upper()}"
            f"  ·  Score: {self.world.score}",
            True,
            YELLOW,
        )
        self.screen.blit(title, (self.PADDING, 8))

        w = self.world
        status = (
            f"Pos {self.agent.pos}  Dir {self.agent.direction.name[:1]}"
            f"  Arrow: {'✓' if w.agent_has_arrow else '✗'}"
            f"  Gold: {'GRABBED!' if self.agent.has_gold else 'not yet'}"
            f"  Step: {self.agent.step_count}"
        )
        s = self.f_body.render(status, True, LIGHT_GRAY)
        self.screen.blit(s, (self.PADDING, 36))

        hint = self.f_small.render(
            "SPACE=pause  →=step  ↑↓=speed  Q=quit", True, GRAY
        )
        self.screen.blit(hint, (self.screen.get_width() - 340, 8))

        if self.paused:
            p = self.f_title.render("⏸ PAUSED", True, ORANGE)
            self.screen.blit(p, (self.screen.get_width() - 340, 30))

    # ------------------------------------------------------------------
    # World grid (actual state, fog of war)
    # ------------------------------------------------------------------

    def _draw_world_grid(self) -> None:
        label = self.f_body.render("ACTUAL WORLD  (fog of war)", True, YELLOW)
        self.screen.blit(label, (self._world_x, self._world_y - 18))

        for r in range(self.world.size):
            for c in range(self.world.size):
                x = self._world_x + c * self.cell_size
                y = self._world_y + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pos = (r, c)
                cell = self.world.grid[r][c]
                is_agent = pos == self.agent.pos
                is_visited = pos in self.agent.kb.visited

                # ── Background ────────────────────────────────────────
                if is_agent:
                    bg = CELL_AGENT
                elif is_visited:
                    if cell.has_pit:
                        bg = CELL_PIT
                    elif cell.has_wumpus and self.world.wumpus_alive:
                        bg = CELL_WUMPUS
                    else:
                        bg = CELL_VISITED
                else:
                    bg = CELL_FOG

                pygame.draw.rect(self.screen, bg, rect)
                pygame.draw.rect(self.screen, BORDER, rect, 1)

                if is_visited or is_agent:
                    self._draw_world_cell_contents(x, y, pos, cell)

                if is_agent:
                    self._draw_agent_arrow(x, y)

                # Coord label
                co = self.f_label.render(f"{r},{c}", True, (75, 80, 95))
                self.screen.blit(co, (x + self.cell_size - 22, y + self.cell_size - 13))

    def _draw_world_cell_contents(self, x: int, y: int, pos, cell) -> None:
        cs = self.cell_size
        symbols: list[tuple[str, tuple]] = []

        if cell.has_stench:
            symbols.append(("S", PURPLE))
        if cell.has_breeze:
            symbols.append(("B", TEAL))
        if cell.has_gold:
            symbols.append(("G", YELLOW))
        if cell.has_pit:
            symbols.append(("PIT", BLUE))
        if cell.has_wumpus and self.world.wumpus_alive:
            symbols.append(("W", RED))
        if (
            not self.world.wumpus_alive
            and self.world.wumpus_pos == pos
        ):
            symbols.append(("☠W", GRAY))

        for i, (sym, color) in enumerate(symbols):
            surf = self.f_small.render(sym, True, color)
            sx = x + 3 + (i % 2) * (cs // 2)
            sy = y + 3 + (i // 2) * (cs // 3)
            self.screen.blit(surf, (sx, sy))

    # ------------------------------------------------------------------
    # Knowledge map
    # ------------------------------------------------------------------

    def _draw_kb_grid(self) -> None:
        label = self.f_body.render("AGENT KNOWLEDGE BASE", True, (105, 200, 255))
        self.screen.blit(label, (self._kb_x, self._kb_y - 18))

        kb = self.agent.kb
        for r in range(self.world.size):
            for c in range(self.world.size):
                x = self._kb_x + c * self.cell_size
                y = self._kb_y + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pos = (r, c)
                status = kb.get_cell_status(pos)
                is_agent = pos == self.agent.pos

                # ── Background ────────────────────────────────────────
                if is_agent:
                    bg = KB_AGENT
                elif status == CellStatus.VISITED:
                    bg = KB_VISITED
                elif status == CellStatus.SAFE:
                    bg = KB_SAFE
                elif status == CellStatus.CONFIRMED_PIT:
                    bg = KB_CONFIRMED_PIT
                elif status == CellStatus.CONFIRMED_WUMPUS:
                    bg = KB_CONFIRMED_WUMPUS
                elif status == CellStatus.POSSIBLY_PIT:
                    bg = KB_PIT_POSSIBLE
                elif status == CellStatus.POSSIBLY_WUMPUS:
                    bg = KB_WUMPUS_POSSIBLE
                elif status == CellStatus.DANGEROUS:
                    bg = KB_DANGEROUS
                else:
                    bg = KB_UNKNOWN

                pygame.draw.rect(self.screen, bg, rect)
                pygame.draw.rect(self.screen, BORDER, rect, 1)

                cs = self.cell_size
                symbols: list[tuple[str, tuple]] = []

                if is_agent:
                    self._draw_agent_arrow(x, y)
                elif pos in kb.visited:
                    percepts = kb.visited[pos]
                    if Perception.STENCH in percepts:
                        symbols.append(("S", PURPLE))
                    if Perception.BREEZE in percepts:
                        symbols.append(("B", TEAL))
                    if Perception.GLITTER in percepts:
                        symbols.append(("G", YELLOW))
                    symbols.append(("✓", LIGHT_GREEN))
                else:
                    if status == CellStatus.CONFIRMED_PIT:
                        symbols.append(("PIT", (150, 200, 255)))
                    elif status == CellStatus.CONFIRMED_WUMPUS:
                        symbols.append(("W!", RED))
                    elif status == CellStatus.POSSIBLY_PIT:
                        symbols.append(("?P", ORANGE))
                    elif status == CellStatus.POSSIBLY_WUMPUS:
                        symbols.append(("?W", RED))
                    elif status == CellStatus.DANGEROUS:
                        symbols.append(("!!", (255, 80, 80)))
                    elif status == CellStatus.SAFE:
                        symbols.append(("safe", LIGHT_GREEN))

                for i, (sym, color) in enumerate(symbols):
                    surf = self.f_small.render(sym, True, color)
                    sx = x + 3 + (i % 2) * (cs // 2)
                    sy = y + 3 + (i // 2) * (cs // 3)
                    self.screen.blit(surf, (sx, sy))

                co = self.f_label.render(f"{r},{c}", True, (65, 70, 85))
                self.screen.blit(co, (x + self.cell_size - 22, y + self.cell_size - 13))

    # ------------------------------------------------------------------
    # Agent arrow
    # ------------------------------------------------------------------

    def _draw_agent_arrow(self, x: int, y: int) -> None:
        cx = x + self.cell_size // 2
        cy = y + self.cell_size // 2
        sz = self.cell_size // 3

        tips = {
            Direction.NORTH: [(cx, cy - sz), (cx - sz // 2, cy + sz // 2), (cx + sz // 2, cy + sz // 2)],
            Direction.SOUTH: [(cx, cy + sz), (cx - sz // 2, cy - sz // 2), (cx + sz // 2, cy - sz // 2)],
            Direction.EAST:  [(cx + sz, cy), (cx - sz // 2, cy - sz // 2), (cx - sz // 2, cy + sz // 2)],
            Direction.WEST:  [(cx - sz, cy), (cx + sz // 2, cy - sz // 2), (cx + sz // 2, cy + sz // 2)],
        }
        pygame.draw.polygon(self.screen, YELLOW, tips[self.agent.direction])

    # ------------------------------------------------------------------
    # Sidebar (status + reasoning log)
    # ------------------------------------------------------------------

    def _draw_sidebar(self) -> None:
        sx = self._side_x
        sy = self._side_y
        sw = self.SIDEBAR_W
        sh = self.screen.get_height() - self.PADDING * 2

        pygame.draw.rect(self.screen, PANEL_BG, (sx, sy, sw, sh))
        pygame.draw.rect(self.screen, BORDER, (sx, sy, sw, sh), 2)

        y = sy + self.PADDING

        # ── Title ─────────────────────────────────────────────────────
        t = self.f_title.render("AI AGENT STATUS", True, (105, 200, 255))
        self.screen.blit(t, (sx + self.PADDING, y))
        y += 26
        self._hline(sx, sx + sw, y)
        y += 8

        # ── KB Statistics ─────────────────────────────────────────────
        kb = self.agent.kb
        rows = [
            ("Visited Cells", str(len(kb.visited)), WHITE),
            ("Confirmed Safe", str(len(kb.safe_cells)), LIGHT_GREEN),
            ("Pit Candidates", str(len(kb.pit_possible)), ORANGE),
            ("Confirmed Pits", str(len(kb.confirmed_pits)), (100, 160, 255)),
            ("Wumpus Candidates", str(len(kb.wumpus_possible)), RED),
            ("Wumpus Dead", "YES ✓" if kb.wumpus_dead else "NO", LIGHT_GREEN if kb.wumpus_dead else RED),
            ("Has Arrow", "YES" if kb.has_arrow else "USED", LIGHT_GREEN if kb.has_arrow else GRAY),
            ("Has Gold", "GRABBED! ✓" if kb.has_gold else "Not yet", YELLOW if kb.has_gold else GRAY),
            ("Score", str(self.world.score), YELLOW),
        ]
        for label, val, col in rows:
            ls = self.f_body.render(label + ":", True, GRAY)
            vs = self.f_body.render(val, True, col)
            self.screen.blit(ls, (sx + self.PADDING, y))
            self.screen.blit(vs, (sx + 185, y))
            y += 18

        y += 4
        self._hline(sx, sx + sw, y)
        y += 6

        # ── Confirmed locations ───────────────────────────────────────
        if kb.confirmed_wumpus and not kb.wumpus_dead:
            s = self.f_body.render(f"⚠ Wumpus confirmed: {kb.confirmed_wumpus}", True, RED)
            self.screen.blit(s, (sx + self.PADDING, y))
            y += 18
        if kb.confirmed_pits:
            ps = ", ".join(str(p) for p in sorted(kb.confirmed_pits)[:4])
            s = self.f_body.render(f"⚠ Confirmed pits: {ps}", True, (100, 160, 255))
            self.screen.blit(s, (sx + self.PADDING, y))
            y += 18

        y += 4
        self._hline(sx, sx + sw, y)
        y += 6

        # ── Legend ────────────────────────────────────────────────────
        lt = self.f_body.render("MAP LEGEND:", True, WHITE)
        self.screen.blit(lt, (sx + self.PADDING, y))
        y += 18

        legend = [
            ("Visited (safe)", KB_VISITED),
            ("Safe (unvisited)", KB_SAFE),
            ("Possibly Pit", KB_PIT_POSSIBLE),
            ("Possibly Wumpus", KB_WUMPUS_POSSIBLE),
            ("Both Possible", KB_DANGEROUS),
            ("Confirmed Pit", KB_CONFIRMED_PIT),
            ("Confirmed Wumpus", KB_CONFIRMED_WUMPUS),
            ("Unknown", KB_UNKNOWN),
        ]
        col1 = legend[: len(legend) // 2 + 1]
        col2 = legend[len(legend) // 2 + 1 :]
        for i, (text, color) in enumerate(col1):
            pygame.draw.rect(self.screen, color, (sx + self.PADDING, y + i * 14 + 2, 10, 10))
            s = self.f_small.render(text, True, LIGHT_GRAY)
            self.screen.blit(s, (sx + self.PADDING + 14, y + i * 14))
        for i, (text, color) in enumerate(col2):
            pygame.draw.rect(self.screen, color, (sx + sw // 2, y + i * 14 + 2, 10, 10))
            s = self.f_small.render(text, True, LIGHT_GRAY)
            self.screen.blit(s, (sx + sw // 2 + 14, y + i * 14))
        y += max(len(col1), len(col2)) * 14 + 4

        self._hline(sx, sx + sw, y)
        y += 6

        # ── AI Reasoning Log ──────────────────────────────────────────
        lt2 = self.f_body.render("AI REASONING LOG  (latest at bottom):", True, YELLOW)
        self.screen.blit(lt2, (sx + self.PADDING, y))
        y += 20

        log = kb.reasoning_log
        line_h = 13
        avail_h = (sy + sh) - y - self.PADDING
        max_lines = avail_h // line_h
        start_idx = max(0, len(log) - max_lines)

        for entry in log[start_idx:]:
            color = self._log_color(entry)
            truncated = entry[:66] if len(entry) > 66 else entry
            s = self.f_small.render(truncated, True, color)
            self.screen.blit(s, (sx + self.PADDING, y))
            y += line_h
            if y + line_h > sy + sh - self.PADDING:
                break

    @staticmethod
    def _log_color(entry: str) -> tuple:
        if entry.startswith("[SAFE]"):
            return LIGHT_GREEN
        if entry.startswith("[BREEZE]") or entry.startswith("[STENCH]"):
            return ORANGE
        if entry.startswith("[INFER]"):
            return (105, 185, 255)
        if entry.startswith("[DECISION]"):
            return YELLOW
        if entry.startswith("[ACTION]"):
            return TEAL
        if entry.startswith("[SCREAM]"):
            return RED
        if entry.startswith("[GLITTER]"):
            return YELLOW
        return LIGHT_GRAY

    def _hline(self, x1: int, x2: int, y: int) -> None:
        pygame.draw.line(self.screen, BORDER, (x1, y), (x2, y))

    # ------------------------------------------------------------------
    # Game-over overlay
    # ------------------------------------------------------------------

    def draw_game_over(self, won: bool, score: int) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2

        if won:
            msg = "🏆 YOU WIN!"
            color = LIGHT_GREEN
        else:
            msg = "💀 GAME OVER"
            color = RED

        big = pygame.font.SysFont("monospace", 48, bold=True)
        mid = pygame.font.SysFont("monospace", 22)
        small = pygame.font.SysFont("monospace", 16)

        ms = big.render(msg, True, color)
        self.screen.blit(ms, (cx - ms.get_width() // 2, cy - 60))

        ss = mid.render(f"Final Score: {score}", True, YELLOW)
        self.screen.blit(ss, (cx - ss.get_width() // 2, cy + 10))

        hs = small.render("Press any key to exit", True, GRAY)
        self.screen.blit(hs, (cx - hs.get_width() // 2, cy + 55))

        pygame.display.flip()
