"""Entry point for the Wumpus World AI Simulation.

Pygame graphical simulation (default when a display is available):
    uv run wumpus-world              # Menu, then 2D simulation window
    uv run wumpus-world --graphical  # Force graphical mode

Terminal-only (headless):
    uv run wumpus-world --headless
    WUMPUS_HEADLESS=1 uv run wumpus-world
"""

from __future__ import annotations

import os
import sys
import time

from wumpus_world.environment import Action, DifficultyLevel, Perception, WumpusWorld
from wumpus_world.agent import Agent


# ---------------------------------------------------------------------------
# Headless / text mode helpers (for CI or non-display environments)
# ---------------------------------------------------------------------------

def _run_headless(difficulty: DifficultyLevel) -> tuple[bool, int]:
    """Run the simulation without a display; print reasoning to stdout."""
    print(f"\n{'='*60}")
    print(f"  WUMPUS WORLD  |  Difficulty: {difficulty.value.upper()}")
    print(f"{'='*60}\n")

    world = WumpusWorld(difficulty)
    agent = Agent(world)

    print(f"  World size : {world.size}×{world.size}")
    print(f"  Gold at    : {world.gold_pos}")
    if world.has_wumpus:
        print(f"  Wumpus at  : {world.wumpus_pos}")
    print(f"  Pits       : {[(r,c) for r in range(world.size) for c in range(world.size) if world.grid[r][c].has_pit]}")
    print()

    max_steps = world.size * world.size * 4
    for _ in range(max_steps):
        if world.game_over:
            break

        action = agent.choose_action()
        print(f"  [{agent.step_count:>3}] Agent at {agent.pos} → {action.name}")

        # Print latest reasoning entries
        kb = agent.kb
        new_entries = kb.reasoning_log[-3:] if kb.reasoning_log else []
        for entry in new_entries:
            print(f"        {entry}")

        result = world.apply_action(action)
        agent.update(result)

    print(f"\n{'='*60}")
    if world.won:
        print("  RESULT: 🏆 Agent WON! Gold retrieved and escaped.")
    elif world.game_over:
        print("  RESULT: 💀 Agent died or gave up.")
    else:
        print("  RESULT: ⏱ Max steps reached without resolution.")
    print(f"  Final score: {world.score}")
    print(f"{'='*60}\n")

    return world.won, world.score


# ---------------------------------------------------------------------------
# Pygame menu
# ---------------------------------------------------------------------------

def _show_menu(screen, fonts: dict) -> DifficultyLevel:
    """Interactive difficulty selection screen."""
    import pygame

    f_title, f_opt, f_desc, f_hint = (
        fonts["title"], fonts["option"], fonts["desc"], fonts["hint"]
    )

    options = [
        (DifficultyLevel.EASY,   "EASY",   "4×4 grid · 2 pits · No Wumpus",           (50, 200, 60)),
        (DifficultyLevel.MEDIUM, "MEDIUM", "6×6 grid · 4 pits · 1 Wumpus",            (230, 150, 40)),
        (DifficultyLevel.HARD,   "HARD",   "8×8 grid · 8 pits · 1 Wumpus",            (215, 55, 55)),
    ]
    selected = 0
    W, H = screen.get_size()

    while True:
        screen.fill((18, 20, 35))

        # Title
        ts = f_title.render("WUMPUS WORLD  AI SIMULATION", True, (252, 220, 0))
        screen.blit(ts, (W // 2 - ts.get_width() // 2, 40))

        sub = f_desc.render("Select difficulty to start:", True, (170, 175, 190))
        screen.blit(sub, (W // 2 - sub.get_width() // 2, 100))

        for i, (level, name, desc, color) in enumerate(options):
            oy = 140 + i * 90
            is_sel = i == selected
            bg = (38, 42, 72) if is_sel else (22, 24, 44)
            pygame.draw.rect(screen, bg, (W // 2 - 240, oy, 480, 72), border_radius=10)
            pygame.draw.rect(
                screen,
                color if is_sel else (50, 55, 90),
                (W // 2 - 240, oy, 480, 72),
                2,
                border_radius=10,
            )
            ns = f_opt.render(name, True, color if is_sel else (140, 145, 180))
            ds = f_desc.render(desc, True, (185, 188, 200))
            screen.blit(ns, (W // 2 - 220, oy + 10))
            screen.blit(ds, (W // 2 - 220, oy + 42))
            if is_sel:
                arrow = f_opt.render("▶", True, color)
                screen.blit(arrow, (W // 2 + 190, oy + 18))

        hs = f_hint.render("↑ ↓ navigate   ENTER select   1/2/3 quick pick   Q quit", True, (95, 100, 120))
        screen.blit(hs, (W // 2 - hs.get_width() // 2, H - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                elif k == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif k == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif k == pygame.K_RETURN:
                    return options[selected][0]
                elif k == pygame.K_1:
                    return DifficultyLevel.EASY
                elif k == pygame.K_2:
                    return DifficultyLevel.MEDIUM
                elif k == pygame.K_3:
                    return DifficultyLevel.HARD


# ---------------------------------------------------------------------------
# Pygame game loop
# ---------------------------------------------------------------------------

def _run_pygame(difficulty: DifficultyLevel) -> tuple[str, bool, int] | tuple[str, DifficultyLevel, int]:
    """Run the full Pygame simulation.

    Returns:
        ("quit", won, score) when the user quits after game over.
        ("play_again", difficulty, score) when the user chooses to play again.
    """
    import pygame
    from wumpus_world.renderer import WumpusRenderer

    world = WumpusWorld(difficulty)
    agent = Agent(world)

    print(f"\n  [START] Difficulty={difficulty.value.upper()}"
          f"  Size={world.size}×{world.size}"
          f"  Gold={world.gold_pos}"
          + (f"  Wumpus={world.wumpus_pos}" if world.has_wumpus else ""))

    renderer = WumpusRenderer(world, agent)
    renderer.draw()

    max_steps = world.size * world.size * 6
    play_again = False

    while renderer.running:
        now = pygame.time.get_ticks()
        ev = renderer.handle_events()

        if world.game_over:
            renderer.draw_game_over(world.won, world.score)
            # Wait for Play again (ENTER/R) or Quit (Q/Esc)
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return ("quit", world.won, world.score)
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            pygame.quit()
                            return ("quit", world.won, world.score)
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r):
                            play_again = True
                            break
                if play_again:
                    break
                renderer.clock.tick(60)
            break

        should_step = (
            (not renderer.paused and now - renderer.last_step_ms >= renderer.step_delay)
            or ev == "STEP"
        )

        if should_step and agent.step_count < max_steps:
            renderer.last_step_ms = now
            action = agent.choose_action()
            result = world.apply_action(action)
            agent.update(result)
            renderer.draw()

        renderer.clock.tick(60)

    if play_again:
        return ("play_again", difficulty, world.score)
    pygame.quit()
    return ("quit", world.won, world.score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    force_graphical = "--graphical" in sys.argv or "-g" in sys.argv
    headless = (
        "--headless" in sys.argv
        or os.environ.get("WUMPUS_HEADLESS", "").lower() in ("1", "true", "yes")
        or (not _has_display() and not force_graphical)
    )

    if headless:
        difficulty = _parse_difficulty_arg()
        won, score = _run_headless(difficulty)
    else:
        import pygame
        pygame.init()
        pygame.font.init()

        W, H = 680, 460
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Wumpus World – Pygame graphical simulation")

        fonts = {
            "title":  pygame.font.SysFont("monospace", 30, bold=True),
            "option": pygame.font.SysFont("monospace", 22, bold=True),
            "desc":   pygame.font.SysFont("monospace", 14),
            "hint":   pygame.font.SysFont("monospace", 12),
        }

        won, score = False, 0
        while True:
            difficulty = _show_menu(screen, fonts)
            result = _run_pygame(difficulty)
            if result[0] == "quit":
                won, score = result[1], result[2]
                break
            # Play again: switch back to menu window size and loop
            screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Wumpus World – Pygame graphical simulation")

        pygame.quit()

    if won:
        print("  Agent successfully retrieved the gold and escaped! 🏆")
    else:
        print("  Better luck next time.")


def _has_display() -> bool:
    """Return True if a display is available."""
    if sys.platform == "win32":
        return True
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    return bool(display or wayland)


def _parse_difficulty_arg() -> DifficultyLevel:
    """Parse --easy / --medium / --hard from argv, default to EASY."""
    for arg in sys.argv[1:]:
        arg_lower = arg.lstrip("-").lower()
        if arg_lower == "easy":
            return DifficultyLevel.EASY
        if arg_lower == "medium":
            return DifficultyLevel.MEDIUM
        if arg_lower == "hard":
            return DifficultyLevel.HARD
    return DifficultyLevel.EASY


if __name__ == "__main__":
    main()
