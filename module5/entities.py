"""
entities.py
Defines the player, level platforms, score collectibles, and moving enemy.
Date: 10/24/2026
Author: Kyle Gortych

Module contract:
    Preconditions: Pygame is installed and constructors receive valid geometry.
    Postconditions: Entity state is represented by Pygame Rect objects.
    Invariants: Rectangles have positive dimensions; the player stays within the
        horizontal world bounds and the enemy stays within its patrol bounds.
    Effects: Entity update methods mutate positions and velocities.
"""

import math
from collections.abc import Mapping, Sequence

import pygame

from constants import (
    GRAVITY,
    JUMP_SPEED,
    MAX_FRAME_TIME,
    PLAYER_HEIGHT,
    PLAYER_SPEED,
    PLAYER_WIDTH,
)


class Platform:
    """A static, rectangular surface that supports player collision."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """Create a platform.

        Preconditions: ``width`` and ``height`` are positive.
        Postconditions: ``rect`` describes the requested platform geometry.
        Invariants: ``rect.width`` and ``rect.height`` remain positive.
        Raises: ValueError if either dimension is not positive.
        Effects: Allocates a Pygame Rect.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Platform width and height must be positive.")
        self.rect = pygame.Rect(x, y, width, height)


class Collectible:
    """A coin that can be collected once for score."""

    SIZE = 18

    def __init__(self, x: int, y: int) -> None:
        """Create a coin centered at ``(x, y)``.

        Preconditions: Coordinates are valid integer pixel positions.
        Postconditions: The coin begins uncollected with a positive-size rect.
        Invariants: ``collected`` is boolean.
        Effects: Allocates a Pygame Rect.
        """
        self.rect = pygame.Rect(x - self.SIZE // 2, y - self.SIZE // 2, self.SIZE, self.SIZE)
        self.collected = False

    def collect(self) -> bool:
        """Mark this coin collected and report whether it changed state.

        Preconditions: None.
        Postconditions: The coin is collected; returns True only on its first
            collection.
        Invariants: A collected coin never becomes uncollected.
        Effects: Mutates ``collected`` on first collection.
        """
        if self.collected:
            return False
        self.collected = True
        return True


class Enemy:
    """A simple enemy that patrols horizontally between two bounds."""

    WIDTH = 34
    HEIGHT = 38

    def __init__(
        self,
        x: int,
        y: int,
        left_bound: int,
        right_bound: int,
        speed: float = 145.0,
    ) -> None:
        """Create a patrol enemy.

        Preconditions: ``left_bound < right_bound`` and speed is positive.
        Postconditions: The enemy begins inside its patrol interval, facing right.
        Invariants: Its left edge remains in
            ``[left_bound, right_bound - rect.width]``.
        Raises: ValueError for invalid bounds or non-positive speed.
        Effects: Allocates a Pygame Rect.
        """
        if left_bound >= right_bound or speed <= 0:
            raise ValueError("Enemy bounds must increase and speed must be positive.")
        if right_bound - left_bound < self.WIDTH:
            raise ValueError("Enemy patrol range must be at least its width.")
        if not left_bound <= x <= right_bound - self.WIDTH:
            raise ValueError("Enemy start position must be within its patrol bounds.")

        self.rect = pygame.Rect(x, y, self.WIDTH, self.HEIGHT)
        self.left_bound = float(left_bound)
        self.right_bound = float(right_bound)
        self.speed = float(speed)
        self.direction = 1
        self._x = float(x)

    def update(self, delta_seconds: float) -> None:
        """Move and bounce the enemy within its patrol interval.

        Preconditions: ``delta_seconds`` is finite and between zero and
            ``MAX_FRAME_TIME``.
        Postconditions: The enemy has moved according to its speed and remains
            within both patrol bounds.
        Invariants: ``direction`` is either -1 or 1.
        Raises: ValueError for an invalid time step.
        Effects: Mutates the enemy position and direction.
        """
        _validate_delta(delta_seconds)
        self._x += self.speed * self.direction * delta_seconds
        if self._x <= self.left_bound:
            self._x = self.left_bound
            self.direction = 1
        elif self._x + self.rect.width >= self.right_bound:
            self._x = self.right_bound - self.rect.width
            self.direction = -1
        self.rect.x = round(self._x)


class Player:
    """A controllable character with gravity, jumping, and platform collision."""

    def __init__(self, x: int, y: int) -> None:
        """Create the player at the given top-left world coordinates.

        Preconditions: Coordinates are valid integer pixel positions.
        Postconditions: The player starts stationary and not grounded.
        Invariants: The rectangle has fixed positive dimensions, and its
            coordinates mirror the internal floating-point position.
        Effects: Allocates a Pygame Rect.
        """
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self._x = float(x)
        self._y = float(y)
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self.on_ground = False

    def update(
        self,
        delta_seconds: float,
        keys: Mapping[int, bool] | Sequence[bool],
        platforms: Sequence[Platform],
        world_width: int,
    ) -> None:
        """Apply input, gravity, and platform collision for one frame.

        Preconditions: ``delta_seconds`` is finite and in range,
            ``world_width`` is at least the player width, and ``keys`` supports
            Pygame key constants.
        Postconditions: Horizontal motion is resolved against the world edges
            and platforms; downward platform contact grounds the player.
        Invariants: The player remains within the horizontal world bounds.
        Raises: ValueError for an invalid time step or world width.
        Effects: Mutates player position, velocity, and grounded state.
        """
        _validate_delta(delta_seconds)
        if world_width < self.rect.width:
            raise ValueError("World width must be at least the player width.")

        left_pressed = _key_pressed(keys, pygame.K_LEFT) or _key_pressed(keys, pygame.K_a)
        right_pressed = _key_pressed(keys, pygame.K_RIGHT) or _key_pressed(keys, pygame.K_d)
        jump_pressed = (
            _key_pressed(keys, pygame.K_SPACE)
            or _key_pressed(keys, pygame.K_UP)
            or _key_pressed(keys, pygame.K_w)
        )

        self._velocity_x = (int(right_pressed) - int(left_pressed)) * PLAYER_SPEED
        self._x += self._velocity_x * delta_seconds
        self._x = min(max(self._x, 0.0), float(world_width - self.rect.width))
        self.rect.x = round(self._x)

        for platform in platforms:
            if self.rect.colliderect(platform.rect):
                if self._velocity_x > 0:
                    self.rect.right = platform.rect.left
                elif self._velocity_x < 0:
                    self.rect.left = platform.rect.right
                self._x = float(self.rect.x)

        if jump_pressed and self.on_ground:
            self._velocity_y = -JUMP_SPEED

        self.on_ground = False
        self._velocity_y += GRAVITY * delta_seconds
        self._y += self._velocity_y * delta_seconds
        self.rect.y = round(self._y)

        for platform in platforms:
            if not self.rect.colliderect(platform.rect):
                continue
            if self._velocity_y > 0:
                self.rect.bottom = platform.rect.top
                self._velocity_y = 0.0
                self.on_ground = True
            elif self._velocity_y < 0:
                self.rect.top = platform.rect.bottom
                self._velocity_y = 0.0
            self._y = float(self.rect.y)


def _key_pressed(keys: Mapping[int, bool] | Sequence[bool], key: int) -> bool:
    """Read a key from either a dictionary or Pygame's pressed-key sequence."""
    if isinstance(keys, Mapping):
        return bool(keys.get(key, False))
    return bool(keys[key])


def _validate_delta(delta_seconds: float) -> None:
    """Reject time steps that could destabilize the physics simulation."""
    if (
        not isinstance(delta_seconds, (int, float))
        or not math.isfinite(delta_seconds)
        or not 0 <= delta_seconds <= MAX_FRAME_TIME
    ):
        raise ValueError(f"delta_seconds must be between 0 and {MAX_FRAME_TIME}.")
