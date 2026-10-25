"""
game.py
Coordinates the level, player, scoring, enemy challenge, and rendering.
Date: 10/24/2026
Author: Kyle Gortych

Module contract:
    Preconditions: Pygame is installed and Game receives a drawable Surface.
    Postconditions: Game state is playing, won, or lost; score equals the
        number of collected coins.
    Invariants: The level geometry is fixed, and terminal states do not change.
    Effects: Updates entity state and draws the complete game frame.
"""

import math
from collections.abc import Mapping, Sequence

import pygame

from constants import (
    BACKGROUND_COLOR,
    COIN_COLOR,
    ENEMY_COLOR,
    FLAG_COLOR,
    GROUND_Y,
    HUD_COLOR,
    MAX_FRAME_TIME,
    PLATFORM_COLOR,
    PLAYER_COLOR,
    SCREEN_HEIGHT,
    TEXT_COLOR,
    WORLD_WIDTH,
)
from entities import Collectible, Enemy, Platform, Player


class Game:
    """Own the platform game state and provide update/draw operations."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Create a level and player for a drawable screen.

        Preconditions: ``screen`` is a Pygame Surface at least 320 by 240.
        Postconditions: A new game begins in the ``playing`` state with score 0.
        Invariants: ``state`` is one of playing, won, or lost.
        Raises: ValueError if the screen is too small.
        Effects: Initializes Pygame's font subsystem if needed and allocates fonts.
        """
        if screen.get_width() < 320 or screen.get_height() < 240:
            raise ValueError("The game screen must be at least 320 by 240 pixels.")

        if not pygame.font.get_init():
            pygame.font.init()

        self.screen = screen
        self.screen_width, self.screen_height = screen.get_size()
        self.player = Player(55, GROUND_Y - 44)
        self.platforms = [
            Platform(0, GROUND_Y, WORLD_WIDTH, SCREEN_HEIGHT - GROUND_Y),
            Platform(230, 450, 170, 20),
            Platform(500, 380, 180, 20),
            Platform(800, 455, 190, 20),
            Platform(1110, 385, 190, 20),
            Platform(1450, 455, 200, 20),
            Platform(1770, 370, 190, 20),
            Platform(2070, 440, 190, 20),
        ]
        self.collectibles = [
            Collectible(285, 414),
            Collectible(555, 344),
            Collectible(870, 419),
            Collectible(1165, 349),
            Collectible(1510, 419),
            Collectible(1830, 334),
            Collectible(2130, 404),
        ]
        self.enemy = Enemy(1370, GROUND_Y - Enemy.HEIGHT, 1330, 1730)
        self.goal_x = WORLD_WIDTH - 65
        self.state = "playing"
        self.score = 0
        self._font = pygame.font.Font(None, 26)
        self._large_font = pygame.font.Font(None, 54)

    def update(
        self,
        delta_seconds: float,
        keys: Mapping[int, bool] | Sequence[bool],
    ) -> None:
        """Advance gameplay by one frame.

        Preconditions: ``delta_seconds`` is finite and non-negative; ``keys``
            supports Pygame key constants.
        Postconditions: Active entities advance by at most ``MAX_FRAME_TIME``;
            collection, collision, falling, and finish conditions update state.
        Invariants: Terminal states and score cannot be changed by later updates.
        Raises: ValueError if ``delta_seconds`` is negative or not finite.
        Effects: Mutates game and entity state.
        """
        if (
            not isinstance(delta_seconds, (int, float))
            or not math.isfinite(delta_seconds)
            or delta_seconds < 0
        ):
            raise ValueError("delta_seconds must be a finite, non-negative number.")
        if self.state != "playing":
            return

        frame_time = min(delta_seconds, MAX_FRAME_TIME)
        self.player.update(frame_time, keys, self.platforms, WORLD_WIDTH)
        self.enemy.update(frame_time)

        for coin in self.collectibles:
            if not coin.collected and self.player.rect.colliderect(coin.rect):
                if coin.collect():
                    self.score += 1

        if self.player.rect.colliderect(self.enemy.rect):
            self.state = "lost"
        elif self.player.rect.top > self.screen_height:
            self.state = "lost"
        elif self.player.rect.right >= self.goal_x:
            self.state = "won"

    def draw(self, screen: pygame.Surface | None = None) -> None:
        """Draw the level, entities, instructions, and terminal-state message.

        Preconditions: The target is a drawable Surface at least 320 by 240.
        Postconditions: The target contains the complete current game frame.
        Invariants: Drawing does not modify gameplay state.
        Raises: ValueError if the target screen is too small.
        Effects: Writes pixels to the target Surface.
        """
        target = screen or self.screen
        if target.get_width() < 320 or target.get_height() < 240:
            raise ValueError("The game screen must be at least 320 by 240 pixels.")

        width, height = target.get_size()
        camera_x = max(0, min(WORLD_WIDTH - width, self.player.rect.centerx - width // 3))
        target.fill(BACKGROUND_COLOR)

        for platform in self.platforms:
            visible_rect = platform.rect.move(-camera_x, 0)
            pygame.draw.rect(target, PLATFORM_COLOR, visible_rect)
            pygame.draw.rect(target, (35, 105, 45), visible_rect, 2)

        for coin in self.collectibles:
            if not coin.collected:
                center = (coin.rect.centerx - camera_x, coin.rect.centery)
                pygame.draw.circle(target, COIN_COLOR, center, coin.rect.width // 2)
                pygame.draw.circle(target, (255, 240, 145), center, coin.rect.width // 2, 2)

        enemy_rect = self.enemy.rect.move(-camera_x, 0)
        pygame.draw.rect(target, ENEMY_COLOR, enemy_rect, border_radius=6)
        player_rect = self.player.rect.move(-camera_x, 0)
        pygame.draw.rect(target, PLAYER_COLOR, player_rect, border_radius=5)

        flag_x = self.goal_x - camera_x
        pygame.draw.line(target, (55, 55, 55), (flag_x, GROUND_Y - 92), (flag_x, GROUND_Y), 5)
        pygame.draw.polygon(
            target,
            FLAG_COLOR,
            [(flag_x + 2, GROUND_Y - 91), (flag_x + 42, GROUND_Y - 78), (flag_x + 2, GROUND_Y - 64)],
        )

        pygame.draw.rect(target, HUD_COLOR, (10, 10, width - 20, 68), border_radius=8)
        score_text = self._font.render(
            f"Coins: {self.score}/{len(self.collectibles)}", True, TEXT_COLOR
        )
        instruction_text = self._font.render(
            "Move: A/D or arrows  |  Jump: W, Space, or Up  |  Reach the flag",
            True,
            TEXT_COLOR,
        )
        target.blit(score_text, (22, 17))
        target.blit(instruction_text, (22, 47))

        if self.state != "playing":
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((10, 18, 32, 175))
            target.blit(overlay, (0, 0))
            message = "You Win!" if self.state == "won" else "Game Over"
            title = self._large_font.render(message, True, TEXT_COLOR)
            details = self._font.render(
                f"Coins: {self.score}/{len(self.collectibles)}    Close the window to exit.",
                True,
                TEXT_COLOR,
            )
            target.blit(title, title.get_rect(center=(width // 2, height // 2 - 22)))
            target.blit(details, details.get_rect(center=(width // 2, height // 2 + 24)))
