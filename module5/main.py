"""
main.py
Entry point for the Module 5 Pygame platform game.
Date: 10/24/2026
Author: Kyle Gortych

Module contract:
    Preconditions: Pygame is installed and a graphical display is available.
    Postconditions: The game window closes cleanly after a quit event.
    Invariants: The game loop updates and draws at no more than FPS frames/sec.
    Effects: Initializes Pygame, opens a window, and runs the game loop.
"""

import pygame

from constants import FPS, SCREEN_HEIGHT, SCREEN_WIDTH
from game import Game


def main() -> None:
    """Initialize Pygame and run the interactive game loop.

    Preconditions: A graphical display is available.
    Postconditions: Pygame is shut down after the window closes.
    Invariants: Each active loop iteration processes events and draws a frame.
    Effects: Opens a window, reads keyboard input, and writes display frames.
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Module 5: Coin Trail")
    clock = pygame.time.Clock()
    game = Game(screen)
    running = True

    try:
        while running:
            delta_seconds = min(clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            game.update(delta_seconds, pygame.key.get_pressed())
            game.draw(screen)
            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
