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

### Code Examples

#### src

<details>
<summary>Click to see</summary>

```python

```
</details>

#### Unit Test

<details>
<summary>Click to see</summary>

```python

```

</details>

#### Integration Test

<details>
<summary>Click to see</summary>

```python

```

</details>

</details>

### Screenshots of Application

<div align="center">
  <img src="screenshots/screenshot_1.png" width="100%" alt="img">
</div>

<div align="center">
  <img src="screenshots/screenshot_2.png" width="100%" alt="img">
</div>
