from typing import Callable

import numpy as np

# Final stage thresholds
POSITION_THRESHOLD = 0.5
ORIENTATION_THRESHOLD = 0.1
LINEAR_VELOCITY_THRESHOLD = 0.5
ANGULAR_VELOCITY_THRESHOLD = 0.1

# Maximum possible accelerations for the ship
LINEAR_ACCELERATION = np.array([2.5, 5.0])
ANGULAR_ACCELERATION = np.array([1.0])


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class LinearController:
    def __init__(self,
                 ship: "Ship"):
        self._ship = ship

    @staticmethod
    def signed_sqrt(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    def derive_acceleration(self) -> np.ndarray:
        ship_rotation = self._ship.rotation

        ship_position = self._ship.position @ ship_rotation.T
        ship_velocity = self._ship.linear_velocity @ ship_rotation.T

        target_position = self._ship.destination @ ship_rotation.T

        position_error = target_position - ship_position
        velocity_error = -ship_velocity

        final_stage = np.abs(position_error) < POSITION_THRESHOLD
        final_stage &= np.abs(velocity_error) < LINEAR_VELOCITY_THRESHOLD

        ideal_product = position_error * LINEAR_ACCELERATION
        ideal_velocity = self.signed_sqrt(2 * ideal_product)

        ideal_weight = np.sign(ideal_velocity - ship_velocity)
        ideal_acceleration = ideal_weight * LINEAR_ACCELERATION

        final_weight = 0.5 * position_error / POSITION_THRESHOLD
        final_weight += 0.5 * velocity_error / LINEAR_VELOCITY_THRESHOLD
        final_acceleration = final_weight * LINEAR_ACCELERATION

        acceleration = np.where(final_stage,
                                final_acceleration,
                                ideal_acceleration)

        return acceleration @ ship_rotation


class AngularController:
    def __init__(self,
                 ship: "Ship"):
        self._ship = ship

    @staticmethod
    def signed_sqrt(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    def derive_acceleration(self) -> np.ndarray:
        ship_position = self._ship.position
        target_position = self._ship.destination

        ship_orientation = self._ship.orientation
        ship_velocity = self._ship.angular_velocity

        position_error = target_position - ship_position
        if np.linalg.norm(position_error) >= POSITION_THRESHOLD:
            target_orientation = np.arctan2(position_error[0],
                                            position_error[1])
        else:
            target_orientation = ship_orientation

        orientation_error = target_orientation - ship_orientation
        orientation_error = wrap_angle(orientation_error)
        velocity_error = -ship_velocity

        final_stage = np.abs(orientation_error) < ORIENTATION_THRESHOLD
        final_stage &= np.abs(velocity_error) < ANGULAR_VELOCITY_THRESHOLD

        ideal_product = orientation_error * ANGULAR_ACCELERATION
        ideal_velocity = self.signed_sqrt(2 * ideal_product)

        ideal_weight = np.sign(ideal_velocity - ship_velocity)
        ideal_acceleration = ideal_weight * ANGULAR_ACCELERATION

        final_weight = 0.5 * orientation_error / ORIENTATION_THRESHOLD
        final_weight += 0.5 * velocity_error / ANGULAR_VELOCITY_THRESHOLD
        final_acceleration = final_weight * ANGULAR_ACCELERATION

        acceleration = np.where(final_stage,
                                final_acceleration,
                                ideal_acceleration)

        return acceleration


class Integrator:
    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 wrap: Callable[[np.ndarray], np.ndarray] | None = None):
        assert position.shape == velocity.shape

        self._position = position.copy()
        self._velocity = velocity.copy()

        self._wrap = wrap

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    def update(self,
               time_step: float,
               acceleration: np.ndarray):
        assert acceleration.shape == self.velocity.shape

        # Semi-implicit Euler method: velocity first
        self._velocity += acceleration * time_step
        self._position += self.velocity * time_step

        if self._wrap is not None:
            self._position = self._wrap(self._position)


class Ship:
    def __init__(self,
                 position: np.ndarray,
                 orientation: np.ndarray,
                 linear_velocity: np.ndarray,
                 angular_velocity: np.ndarray):
        self._linear_motion = Integrator(position,
                                         linear_velocity)
        self._angular_motion = Integrator(orientation,
                                          angular_velocity,
                                          wrap_angle)

        self._linear_control = LinearController(self)
        self._angular_control = AngularController(self)

        self.destination = position

    @property
    def position(self) -> np.ndarray:
        return self._linear_motion.position

    @property
    def orientation(self) -> np.ndarray:
        return self._angular_motion.position

    @property
    def rotation(self) -> np.ndarray:
        angle = self._angular_motion.position[0]
        sin_angle, cos_angle = np.sin(angle), np.cos(angle)
        return np.array([[cos_angle, -sin_angle],
                         [sin_angle, cos_angle]])

    @property
    def linear_velocity(self) -> np.ndarray:
        return self._linear_motion.velocity

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._angular_motion.velocity

    @property
    def destination(self) -> np.ndarray:
        return self._destination

    @destination.setter
    def destination(self, position: np.ndarray):
        assert position.shape == self.position.shape

        self._destination = position.copy()

    def update(self, time_step: float):
        linear_acceleration = self._linear_control.derive_acceleration()
        angular_acceleration = self._angular_control.derive_acceleration()

        self._linear_motion.update(time_step, linear_acceleration)
        self._angular_motion.update(time_step, angular_acceleration)
