import numpy as np

from . import ship


class World:
    def __init__(self):
        self.ships: list[ship.Ship] = []

    def update(self, time_step: float):
        for ship_ in self.ships:
            ship_.update(time_step)
