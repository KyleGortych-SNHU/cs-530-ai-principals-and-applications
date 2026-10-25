<div align="right">
 
![GitHub Action Badge](https://img.shields.io/github/actions/workflow/status/KyleGortych-SNHU/cs-530-ai-principals-and-applications/main.yml?label=main)

</div>

# cs-530-ai-principals-and-applications
Work from AI Principals and Applications

## About

## Motivation
To show the work done in CS 510 Operating Systems Principles from SNHU.

## Getting Started
Setup is done via `git clone url`.

Then the dependecies to add are .

## Installation

### Tools
Installation best via a system level package manager or ephemeral build environment.
Transitvie dependecies such as language are shown via tree level.

- git
- python 3.14

## Reflection


## Design Overview

Testing is done using main.yml as main entrypoint to run the diffrent project tests.

### Module 5

<details>
<summary>Click to see</summary>

#### UML Class Diagram

```mermaid
classDiagram
    class Game {
        -Player player
        -list~Platform~ platforms
        -list~Collectible~ collectibles
        -Enemy enemy
        -int goal_x
        -str state
        -int score
        +update(delta_seconds, keys)
        +draw(screen)
    }
    class Player {
        -Rect rect
        -float velocity_y
        -bool on_ground
        +update(delta_seconds, keys, platforms, world_width)
    }
    class Platform {
        +Rect rect
    }
    class Collectible {
        +Rect rect
        +bool collected
        +collect()
    }
    class Enemy {
        +Rect rect
        +float speed
        +int direction
        +update(delta_seconds)
    }
    Game *-- Player : controls
    Game *-- Platform : level geometry
    Game *-- Collectible : tracks score
    Game *-- Enemy : checks collisions
```

#### Source Code

- [Game entry point](module5/main.py)
- [Game state and rendering](module5/game.py)
- [Player, platform, collectible, and enemy entities](module5/entities.py)
- [Game constants](module5/constants.py)
- [Python dependencies](module5/requirements.txt)

Run the game from the repository root:

```bash
python module5/main.py
```

#### Testing

Run the unit tests from the repository root:

```bash
python -m pytest module5
```

The tests in [test_game.py](module5/test_game.py) cover player jumping, platform
landing, world boundaries, entity validation, coin scoring, enemy collision,
falling, win/loss states, and rendering to an in-memory Pygame surface. GitHub
Actions runs this same test command on pushes to `main` and on manual dispatch.

#### Screenshots

<div align="center">
  <img src="module5/assets/Screenshot_20261025_161016.png" width="90%" alt="Coin Trail gameplay showing the player, platforms, a coin, and an enemy">
  <p>Gameplay</p>
</div>

<div align="center">
  <img src="module5/assets/Screenshot_20261025_161030.png" width="90%" alt="Coin Trail game over screen after an enemy collision">
  <p>Game over screen</p>
</div>

<div align="center">
  <img src="module5/assets/Screenshot_20261025_161118.png" width="90%" alt="Terminal showing the original seven Module 5 tests passing">
  <p>Initial test run</p>
</div>

<div align="center">
  <img src="module5/assets/Screenshot_20261025_161905.png" width="90%" alt="Terminal showing all 17 Module 5 tests passing">
  <p>Updated test run</p>
</div>
