"""
test_game.py
Unit tests for game entities, scoring, and terminal game states.
Date: 10/25/2026
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
from entities import Collectible, Enemy, Platform, Player
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

    def test_player_lands_on_platform(self) -> None:
        """Downward movement stops at the top of a platform."""
        player = Player(20, 55)
        platform = Platform(0, 100, 150, 20)

        player.update(0.05, {}, [platform], world_width=200)

        self.assertEqual(player.rect.bottom, platform.rect.top)
        self.assertTrue(player.on_ground)

    def test_player_stays_inside_world_horizontal_bounds(self) -> None:
        """Left and right input cannot move the player beyond the world."""
        player = Player(0, 100)

        player.update(0.05, {pygame.K_LEFT: True}, [], world_width=100)
        self.assertEqual(player.rect.left, 0)

        right_edge_player = Player(68, 100)
        right_edge_player.update(0.05, {pygame.K_RIGHT: True}, [], world_width=100)
        self.assertEqual(right_edge_player.rect.right, 100)

    def test_player_rejects_world_narrower_than_player(self) -> None:
        """An impossible world width is rejected before movement."""
        player = Player(0, 100)

        with self.assertRaises(ValueError):
            player.update(0, {}, [], world_width=player.rect.width - 1)

    def test_enemy_rejects_invalid_patrol_range(self) -> None:
        """Enemy patrol bounds must contain its full width."""
        with self.assertRaises(ValueError):
            Enemy(0, 0, 0, Enemy.WIDTH - 1)

    def test_platform_rejects_non_positive_dimensions(self) -> None:
        """Platforms with zero or negative dimensions are rejected."""
        for dimensions in ((0, 20), (20, 0), (-1, 20), (20, -1)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    Platform(0, 0, *dimensions)


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

    def test_enemy_collision_causes_loss(self) -> None:
        """Contact with the enemy transitions the game to its lost state."""
        self.game.enemy.rect.center = self.game.player.rect.center

        self.game.update(0, {})

        self.assertEqual(self.game.state, "lost")

    def test_falling_below_screen_causes_loss(self) -> None:
        """A player below the visible world transitions to the lost state."""
        self.game.platforms = []

        for _ in range(30):
            self.game.update(0.05, {})
            if self.game.state == "lost":
                break

        self.assertEqual(self.game.state, "lost")

    def test_terminal_state_is_unchanged_by_later_updates(self) -> None:
        """A lost state does not resume or change after another update."""
        self.game.state = "lost"
        previous_position = self.game.player.rect.copy()

        self.game.update(0.05, {pygame.K_RIGHT: True})

        self.assertEqual(self.game.state, "lost")
        self.assertEqual(self.game.player.rect, previous_position)

    def test_draw_renders_to_in_memory_surface(self) -> None:
        """Drawing updates a Surface and does not require a display window."""
        target = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.game.draw(target)

        self.assertEqual(target.get_size(), (SCREEN_WIDTH, SCREEN_HEIGHT))

    def test_game_rejects_surface_that_is_too_small(self) -> None:
        """Game initialization requires a minimum drawable screen size."""
        with self.assertRaises(ValueError):
            Game(pygame.Surface((100, 100)))

    def test_draw_rejects_surface_that_is_too_small(self) -> None:
        """Rendering rejects an undersized target Surface."""
        with self.assertRaises(ValueError):
            self.game.draw(pygame.Surface((100, 100)))


if __name__ == "__main__":
    unittest.main()
