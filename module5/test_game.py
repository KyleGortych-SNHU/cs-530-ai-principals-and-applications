"""
test_game.py
Unit tests for game entities, scoring, and terminal game states.
Date: 10/24/2026
Author: Kyle Gortych

Module contract:
    Preconditions: Pygame and the Module 5 game modules are importable.
    Postconditions: Tests verify entity and game-state behavior without a display.
    Invariants: Each test creates isolated game state.
    Effects: Allocates in-memory Pygame Surfaces; does not open a window.
"""

import unittest

import pygame

from constants import SCREEN_HEIGHT, SCREEN_WIDTH
from entities import Collectible, Enemy, Player
from game import Game


class EntityTests(unittest.TestCase):
    """Verify entity state transitions and movement constraints."""

    def test_collectible_only_scores_once(self) -> None:
        """Repeated collection reports one state change and stays collected."""
        coin = Collectible(20, 30)

        self.assertTrue(coin.collect())
        self.assertFalse(coin.collect())
        self.assertTrue(coin.collected)

    def test_enemy_bounces_at_patrol_bound(self) -> None:
        """Enemy movement reverses before crossing its right patrol limit."""
        enemy = Enemy(66, 0, 0, 100, speed=100.0)

        enemy.update(0.05)

        self.assertEqual(enemy.rect.right, 100)
        self.assertEqual(enemy.direction, -1)

    def test_player_jumps_when_grounded(self) -> None:
        """A grounded player responds to jump input with upward movement."""
        player = Player(10, 100)
        player.on_ground = True

        player.update(0.01, {pygame.K_SPACE: True}, [], world_width=100)

        self.assertLess(player.rect.y, 100)
        self.assertFalse(player.on_ground)


class GameTests(unittest.TestCase):
    """Verify scoring and terminal-state behavior in a headless Surface."""

    def setUp(self) -> None:
        """Create isolated game state backed by an in-memory Surface."""
        pygame.font.init()
        self.game = Game(pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)))

    def test_overlapping_coin_increases_score_once(self) -> None:
        """Collecting a coin increments score once and removes it from play."""
        coin = self.game.collectibles[0]
        coin.rect.center = self.game.player.rect.center

        self.game.update(0, {})
        self.game.update(0, {})

        self.assertEqual(self.game.score, 1)
        self.assertTrue(coin.collected)

    def test_reaching_finish_wins_and_later_updates_do_not_change_state(self) -> None:
        """Finish contact wins and terminal state remains unchanged."""
        self.game.goal_x = self.game.player.rect.right - 1

        self.game.update(0, {})
        self.game.update(0, {})

        self.assertEqual(self.game.state, "won")

    def test_update_rejects_negative_time(self) -> None:
        """Negative elapsed time is rejected without changing the game state."""
        with self.assertRaises(ValueError):
            self.game.update(-0.01, {})

        self.assertEqual(self.game.state, "playing")

    def test_draw_renders_to_in_memory_surface(self) -> None:
        """Drawing updates a Surface and does not require a display window."""
        target = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.game.draw(target)

        self.assertEqual(target.get_size(), (SCREEN_WIDTH, SCREEN_HEIGHT))


if __name__ == "__main__":
    unittest.main()
