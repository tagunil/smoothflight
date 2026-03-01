import sys

import numpy as np
import pygame

from . import ship
from . import world


# Frame rate per second
FRAME_RATE = 50

# Simulation rate per second
STEP_RATE = 1000

# Scale between world and screen
DISPLAY_SCALE = 10

# Visible world and screen dimensions
WORLD_SIZE = np.array([100.0, 100.0])
SCREEN_SIZE = WORLD_SIZE * DISPLAY_SCALE

# Axis direction remapping
AXIS_REMAP = np.array([1.0, -1.0])

# Left mouse button index
MOUSE_BUTTON_LEFT = 1

# Close range thresholds
POSITION_THRESHOLD = 0.1
VELOCITY_THRESHOLD = 0.1


def close_approach(destination: np.ndarray,
                   position: np.ndarray,
                   velocity: np.ndarray) -> bool:
    nearby = np.linalg.norm(destination - position) < POSITION_THRESHOLD
    nearby = nearby and np.linalg.norm(velocity) < VELOCITY_THRESHOLD
    return nearby


def world_to_screen(position: np.ndarray) -> np.ndarray:
    return SCREEN_SIZE / 2 + position * DISPLAY_SCALE * AXIS_REMAP


def screen_to_world(position: np.ndarray) -> np.ndarray:
    return (position / DISPLAY_SCALE - WORLD_SIZE / 2) * AXIS_REMAP


def draw_ship(screen: pygame.Surface,
              position: np.ndarray,
              rotation: np.ndarray):
    polygon = np.array([[0.0, 2.0],
                        [1.0, -2.0],
                        [0.0, -1.0],
                        [-1.0, -2.0]])

    polygon @= rotation
    polygon += position

    pygame.draw.polygon(screen,
                        pygame.Color("white"),
                        world_to_screen(polygon))


def draw_target(screen: pygame.Surface,
                position: np.ndarray):
    line_1 = np.array([[-0.75, -0.75],
                       [0.75, 0.75]])
    line_2 = np.array([[-0.75, 0.75],
                       [0.75, -0.75]])

    line_1 += position
    line_2 += position

    pygame.draw.line(screen,
                     pygame.Color("white"),
                     world_to_screen(line_1[0]),
                     world_to_screen(line_1[1]),
                     3)
    pygame.draw.line(screen,
                     pygame.Color("white"),
                     world_to_screen(line_2[0]),
                     world_to_screen(line_2[1]),
                     3)


def main() -> int:
    _world = world.World()

    _ship = ship.Ship(np.array([0.0, 0.0]),
                      np.array([0.0]),
                      np.array([0.0, 0.0]),
                      np.array([0.0]))
    _world.ships.append(_ship)

    destination = np.array([0.0, 0.0])

    pygame.init()

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == MOUSE_BUTTON_LEFT:
                    destination = screen_to_world(np.array(event.pos))
                    _ship.destination = destination

        screen.fill(pygame.Color("black"))

        draw_ship(screen,
                  _ship.position,
                  _ship.rotation)

        if not close_approach(destination,
                              _ship.position,
                              _ship.linear_velocity):
            draw_target(screen, destination)

        pygame.display.flip()

        for step in range(STEP_RATE // FRAME_RATE):
            _world.update(1.0 / STEP_RATE)

        clock.tick(FRAME_RATE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
