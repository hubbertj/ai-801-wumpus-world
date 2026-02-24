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
from pathlib import Path
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
        pygame.display.set_caption("Wumpus World – Pygame graphical simulation")

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

        # Agent sprite sheet (2×2: front, back, left, right); keep ref so subsurfaces stay valid
        self._agent_sprite_sheet: Optional[pygame.Surface] = None
        self._agent_sprites: Optional[dict[Direction, pygame.Surface]] = self._load_agent_sprite_sheet()

        # Wumpus character image (optional)
        self._wumpus_sprite: Optional[pygame.Surface] = self._load_wumpus_sprite()

        # Pit image (optional)
        self._pit_sprite: Optional[pygame.Surface] = self._load_pit_sprite()

        # Arrow image (weapon used to kill the Wumpus)
        self._arrow_sprite: Optional[pygame.Surface] = self._load_arrow_sprite()

        # Gold / treasure chest image (optional)
        self._gold_sprite: Optional[pygame.Surface] = self._load_gold_sprite()

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

        # Arrow icon (weapon to kill Wumpus) in header
        if self._arrow_sprite is not None:
            aw, ah = 40, 20
            arr = pygame.transform.smoothscale(self._arrow_sprite, (aw, ah))
            if not w.agent_has_arrow:
                arr.set_alpha(120)
            self.screen.blit(arr, (self.PADDING + 245, 32))

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

                # Cell shading for 2D tile look
                self._draw_cell_shading(rect, bg)
                pygame.draw.rect(self.screen, BORDER, rect, 1)

                if is_visited or is_agent:
                    self._draw_world_cell_contents(x, y, pos, cell)

                if is_agent:
                    self._draw_agent_sprite(x, y)

                # Coord label
                co = self.f_label.render(f"{r},{c}", True, (75, 80, 95))
                self.screen.blit(co, (x + self.cell_size - 22, y + self.cell_size - 13))

    def _draw_cell_shading(self, rect: pygame.Rect, base_color: tuple) -> None:
        """Add subtle highlight/shadow so cells look like 2D tiles."""
        if rect.width < 20:
            return
        # Top/left highlight (lighter)
        highlight = tuple(min(255, c + 18) for c in base_color)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 1)
        # Bottom/right shadow
        shadow = tuple(max(0, c - 15) for c in base_color)
        pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright, 1)
        pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright, 1)

    def _draw_world_cell_contents(self, x: int, y: int, pos, cell) -> None:
        cs = self.cell_size
        cx, cy = x + cs // 2, y + cs // 2
        margin = max(4, cs // 6)

        # Draw 2D graphics for hazards and gold (actual world view)
        if cell.has_pit:
            self._draw_pit_graphic(x, y, cs)
        if cell.has_wumpus and self.world.wumpus_alive:
            self._draw_wumpus_graphic(x, y, cs)
        if not self.world.wumpus_alive and self.world.wumpus_pos == pos:
            self._draw_wumpus_dead_graphic(x, y, cs)
        if cell.has_gold:
            self._draw_gold_graphic(x, y, cs)

        # Percept icons (small, so they don't overlap main graphics)
        icon_y = y + cs - margin - 10
        if cell.has_stench:
            self._draw_stench_icon(x + margin, icon_y)
        if cell.has_breeze:
            self._draw_breeze_icon(x + cs // 2 - 6, icon_y)
        if cell.has_breeze and cell.has_stench:
            self._draw_breeze_icon(x + cs - margin - 12, icon_y)

    def _draw_pit_graphic(self, x: int, y: int, cs: int) -> None:
        """Draw a pit using the pit image, or fallback to a dark hole."""
        if self._pit_sprite is not None:
            sw, sh = self._pit_sprite.get_size()
            size = min(cs - 4, sw, sh)
            if size > 0:
                scale = size / max(sw, sh)
                nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
                scaled = pygame.transform.smoothscale(self._pit_sprite, (nw, nh))
                dx = x + (cs - nw) // 2
                dy = y + (cs - nh) // 2
                self.screen.blit(scaled, (dx, dy))
            return
        # Fallback: dark hole with inner shadow
        cx, cy = x + cs // 2, y + cs // 2
        r = min(cs // 3, 20)
        pygame.draw.circle(self.screen, (5, 5, 25), (cx, cy), r)
        pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), r - 2)
        pygame.draw.ellipse(
            self.screen, (20, 25, 50),
            (cx - r, cy - r // 2, r * 2, r),
            1
        )

    def _load_pit_sprite(self) -> Optional[pygame.Surface]:
        """Load pit image from assets. Returns None on failure."""
        try:
            path = Path(__file__).resolve().parent / "assets" / "pit.png"
            if not path.is_file():
                return None
            surf = pygame.image.load(str(path)).convert_alpha()
            return surf if surf.get_width() and surf.get_height() else None
        except Exception:
            return None

    def _load_arrow_sprite(self) -> Optional[pygame.Surface]:
        """Load arrow image (weapon used to kill the Wumpus). Returns None on failure."""
        try:
            path = Path(__file__).resolve().parent / "assets" / "arrow.png"
            if not path.is_file():
                return None
            surf = pygame.image.load(str(path)).convert_alpha()
            return surf if surf.get_width() and surf.get_height() else None
        except Exception:
            return None

    def _load_gold_sprite(self) -> Optional[pygame.Surface]:
        """Load gold / treasure chest image from assets. Returns None on failure."""
        try:
            path = Path(__file__).resolve().parent / "assets" / "gold.png"
            if not path.is_file():
                return None
            surf = pygame.image.load(str(path)).convert_alpha()
            return surf if surf.get_width() and surf.get_height() else None
        except Exception:
            return None

    def _load_wumpus_sprite(self) -> Optional[pygame.Surface]:
        """Load Wumpus character image from assets. Returns None on failure."""
        try:
            path = Path(__file__).resolve().parent / "assets" / "wumpus.png"
            if not path.is_file():
                return None
            surf = pygame.image.load(str(path)).convert_alpha()
            return surf if surf.get_width() and surf.get_height() else None
        except Exception:
            return None

    def _draw_wumpus_graphic(self, x: int, y: int, cs: int) -> None:
        """Draw the Wumpus using the character image, or fallback to simple face."""
        cx, cy = x + cs // 2, y + cs // 2
        if self._wumpus_sprite is not None:
            sw, sh = self._wumpus_sprite.get_size()
            size = min(cs - 4, sw, sh)
            if size > 0:
                scale = size / max(sw, sh)
                nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
                scaled = pygame.transform.smoothscale(self._wumpus_sprite, (nw, nh))
                dx = x + (cs - nw) // 2
                dy = y + (cs - nh) // 2
                self.screen.blit(scaled, (dx, dy))
            return
        # Fallback: simple monster face
        r = min(cs // 3, 18)
        pygame.draw.circle(self.screen, DARK_RED, (cx, cy), r)
        pygame.draw.circle(self.screen, RED, (cx, cy), r, 2)
        eye_off = r // 2
        pygame.draw.circle(self.screen, YELLOW, (cx - eye_off, cy - 3), 3)
        pygame.draw.circle(self.screen, YELLOW, (cx + eye_off, cy - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (cx - eye_off, cy - 3), 1)
        pygame.draw.circle(self.screen, BLACK, (cx + eye_off, cy - 3), 1)

    def _draw_wumpus_dead_graphic(self, x: int, y: int, cs: int) -> None:
        """Draw dead Wumpus (X eyes, gray)."""
        cx, cy = x + cs // 2, y + cs // 2
        r = min(cs // 3, 18)
        pygame.draw.circle(self.screen, (50, 45, 45), (cx, cy), r)
        pygame.draw.line(self.screen, GRAY, (cx - r, cy), (cx + r, cy), 2)
        pygame.draw.line(self.screen, GRAY, (cx - 5, cy - 5), (cx + 5, cy + 5), 1)
        pygame.draw.line(self.screen, GRAY, (cx + 5, cy - 5), (cx - 5, cy + 5), 1)

    def _draw_gold_graphic(self, x: int, y: int, cs: int) -> None:
        """Draw gold using the treasure chest image, or fallback to shiny coin."""
        if self._gold_sprite is not None:
            sw, sh = self._gold_sprite.get_size()
            size = min(cs - 4, sw, sh)
            if size > 0:
                scale = size / max(sw, sh)
                nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
                scaled = pygame.transform.smoothscale(self._gold_sprite, (nw, nh))
                dx = x + (cs - nw) // 2
                dy = y + (cs - nh) // 2
                self.screen.blit(scaled, (dx, dy))
            return
        # Fallback: shiny pile/coin
        cx, cy = x + cs // 2, y + cs // 2 - 2
        r = min(8, cs // 4)
        pygame.draw.circle(self.screen, (220, 180, 0), (cx, cy), r)
        pygame.draw.circle(self.screen, YELLOW, (cx, cy), r - 1)
        pygame.draw.circle(self.screen, (255, 235, 120), (cx - 2, cy - 2), 2)

    def _draw_stench_icon(self, sx: int, sy: int) -> None:
        """Small wavy stench lines."""
        for i in range(3):
            pygame.draw.arc(
                self.screen, PURPLE,
                (sx + i * 4, sy, 8, 8), 0, 3.14, 1
            )

    def _draw_breeze_icon(self, sx: int, sy: int) -> None:
        """Small wind lines."""
        for i in range(3):
            pygame.draw.line(
                self.screen, TEAL,
                (sx + i * 4, sy + 6), (sx + i * 4 + 3, sy), 1
            )

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

                self._draw_cell_shading(rect, bg)
                pygame.draw.rect(self.screen, BORDER, rect, 1)

                cs = self.cell_size
                symbols: list[tuple[str, tuple]] = []

                if is_agent:
                    self._draw_agent_sprite(x, y)
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
                        self._draw_pit_graphic(x, y, cs)
                    elif status == CellStatus.CONFIRMED_WUMPUS:
                        self._draw_wumpus_graphic(x, y, cs)
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
    # Agent sprite (sprite sheet or fallback pixel-art)
    # ------------------------------------------------------------------

    def _load_agent_sprite_sheet(self) -> Optional[dict[Direction, pygame.Surface]]:
        """Load 2×2 agent sprite sheet (front, back, left, right). Returns None on failure."""
        try:
            assets_dir = Path(__file__).resolve().parent / "assets"
            path = assets_dir / "agent_sprite.png"
            if not path.is_file():
                return None
            sheet = pygame.image.load(str(path))
            sheet = sheet.convert_alpha()
            w, h = sheet.get_width(), sheet.get_height()
            if w < 2 or h < 2:
                return None
            self._agent_sprite_sheet = sheet
            tw, th = w // 2, h // 2
            # Layout: top-left=Front(SOUTH), top-right=Back(NORTH), bottom-left=Left(WEST), bottom-right=Right(EAST)
            sprites = {
                Direction.SOUTH: sheet.subsurface((0, 0, tw, th)),
                Direction.NORTH: sheet.subsurface((tw, 0, tw, th)),
                Direction.WEST: sheet.subsurface((0, th, tw, th)),
                Direction.EAST: sheet.subsurface((tw, th, tw, th)),
            }
            return sprites
        except Exception:
            return None

    # Fallback pixel-art when no sprite sheet
    _AGENT_PIXELS = [
        (3, 0, "H"), (4, 0, "H"),
        (2, 1, "H"), (3, 1, "H"), (4, 1, "H"), (5, 1, "H"),
        (3, 2, "H"), (4, 2, "H"),
        (2, 3, "B"), (3, 3, "B"), (4, 3, "B"), (5, 3, "B"),
        (2, 4, "B"), (3, 4, "B"), (4, 4, "B"), (5, 4, "B"),
        (2, 5, "B"), (3, 5, "B"), (4, 5, "B"), (5, 5, "B"),
        (3, 6, "L"), (4, 6, "L"),
        (3, 7, "L"), (4, 7, "L"),
    ]
    _AGENT_COLORS = {
        "H": (255, 220, 180),
        "B": (28, 75, 195),
        "L": (50, 45, 70),
    }

    def _draw_agent_sprite(self, x: int, y: int) -> None:
        """Draw the agent using the sprite sheet (or pixel-art fallback)."""
        cs = self.cell_size
        direction = self.agent.direction

        if self._agent_sprites is not None and direction in self._agent_sprites:
            sprite = self._agent_sprites[direction]
            sw, sh = sprite.get_size()
            scale = min((cs - 4) / sw, (cs - 4) / sh, 1.0) if sw and sh else 1.0
            nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
            scaled = pygame.transform.smoothscale(sprite, (nw, nh))
            dx = x + (cs - nw) // 2
            dy = y + (cs - nh) // 2
            self.screen.blit(scaled, (dx, dy))
            return

        # Fallback: pixel-art person
        pw = max(2, (cs - 8) // 8)
        ph = max(2, (cs - 8) // 10)
        ox = x + (cs - 8 * pw) // 2
        oy = y + (cs - 10 * ph) // 2
        for px, py, key in self._AGENT_PIXELS:
            rect = pygame.Rect(ox + px * pw, oy + py * ph, pw + 1, ph + 1)
            pygame.draw.rect(self.screen, self._AGENT_COLORS[key], rect)
        arr = max(2, min(4, cs // 12))
        cx, cy = x + cs // 2, y + cs // 2
        arrows = {
            Direction.NORTH: [(cx, oy - 1), (cx - arr, oy + arr), (cx + arr, oy + arr)],
            Direction.SOUTH: [(cx, oy + 10 * ph + 1), (cx - arr, oy + 10 * ph - arr), (cx + arr, oy + 10 * ph - arr)],
            Direction.EAST:  [(ox + 8 * pw + 1, cy), (ox + 8 * pw - arr, cy - arr), (ox + 8 * pw - arr, cy + arr)],
            Direction.WEST:  [(ox - 1, cy), (ox + arr, cy - arr), (ox + arr, cy + arr)],
        }
        pygame.draw.polygon(self.screen, YELLOW, arrows[direction])
        pygame.draw.polygon(self.screen, (255, 255, 150), arrows[direction], 1)

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
            # Arrow icon next to "Has Arrow" row (weapon to kill Wumpus)
            if label == "Has Arrow" and self._arrow_sprite is not None:
                aw, ah = 24, 12
                arr = pygame.transform.smoothscale(self._arrow_sprite, (aw, ah))
                if not kb.has_arrow:
                    arr = arr.copy()
                    arr.set_alpha(120)
                self.screen.blit(arr, (sx + 185 + vs.get_width() + 6, y - 1))
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

        play_again = small.render("ENTER or R = Play again", True, LIGHT_GREEN)
        quit_hint = small.render("Q = Quit", True, GRAY)
        self.screen.blit(play_again, (cx - play_again.get_width() // 2, cy + 48))
        self.screen.blit(quit_hint, (cx - quit_hint.get_width() // 2, cy + 66))

        pygame.display.flip()
