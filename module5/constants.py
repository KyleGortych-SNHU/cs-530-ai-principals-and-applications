"""
constants.py
Shared display, world, physics, and color settings for the platform game.
Date: 10/24/2026
Author: Kyle Gortych

Module contract:
    Preconditions: None.
    Postconditions: Exposes immutable-by-convention settings used by game modules.
    Invariants: Dimensions and gameplay limits are positive; colors are RGB tuples.
    Effects: None.
"""

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WORLD_WIDTH = 2400
FPS = 60

GROUND_Y = 540
PLAYER_WIDTH = 32
PLAYER_HEIGHT = 44
PLAYER_SPEED = 280.0
JUMP_SPEED = 620.0
GRAVITY = 1650.0
MAX_FRAME_TIME = 0.05

BACKGROUND_COLOR = (135, 206, 235)
PLATFORM_COLOR = (56, 142, 60)
PLAYER_COLOR = (40, 90, 220)
COIN_COLOR = (255, 205, 35)
ENEMY_COLOR = (200, 55, 55)
HUD_COLOR = (25, 35, 55)
TEXT_COLOR = (255, 255, 255)
FLAG_COLOR = (235, 70, 65)
