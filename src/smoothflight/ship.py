import numpy as np

# Final stage thresholds
POSITION_THRESHOLD = 0.1
LINEAR_VELOCITY_THRESHOLD = 0.1

# Maximum possible accelerations for the ship
LINEAR_ACCELERATION = np.array([2.5, 5.0])


class LinearController:
    def __init__(self,
                 parent: "Ship",
                 target: np.ndarray):
        self._parent = parent
        self.target = target

    @property
    def target(self) -> np.ndarray:
        return self._target

    @target.setter
    def target(self, position: np.ndarray):
        assert position.shape == self._parent.position.shape

        self._target = position.copy()

    @staticmethod
    def signed_sqrt(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    def derive_acceleration(self) -> np.ndarray:
        parent_rotation = self._parent.rotation

        parent_position = self._parent.position @ parent_rotation.T
        parent_velocity = self._parent.linear_velocity @ parent_rotation.T

        target_position = self._target @ parent_rotation.T

        position_error = target_position - parent_position
        velocity_error = -parent_velocity

        final_stage = np.abs(position_error) < POSITION_THRESHOLD
        final_stage &= np.abs(velocity_error) < LINEAR_VELOCITY_THRESHOLD

        ideal_product = position_error * LINEAR_ACCELERATION
        ideal_velocity = self.signed_sqrt(2 * ideal_product)

        ideal_weight = np.sign(ideal_velocity - parent_velocity)
        ideal_acceleration = ideal_weight * LINEAR_ACCELERATION

        final_weight = 0.5 * position_error / POSITION_THRESHOLD
        final_weight += 0.5 * velocity_error / LINEAR_VELOCITY_THRESHOLD
        final_acceleration = final_weight * LINEAR_ACCELERATION

        acceleration = np.where(final_stage,
                                final_acceleration,
                                ideal_acceleration)

        return acceleration @ parent_rotation


class Integrator:
    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray):
        assert position.shape == velocity.shape

        self.position = position.copy()
        self.velocity = velocity.copy()

    def update(self,
               time_step: float,
               acceleration: np.ndarray):
        assert acceleration.shape == self.velocity.shape

        # Semi-implicit Euler method: velocity first
        self.velocity += acceleration * time_step
        self.position += self.velocity * time_step


class Ship:
    def __init__(self,
                 position: np.ndarray,
                 orientation: np.ndarray,
                 linear_velocity: np.ndarray,
                 angular_velocity: np.ndarray):
        self._linear_motion = Integrator(position, linear_velocity)
        self._angular_motion = Integrator(orientation, angular_velocity)

        self._linear_control = LinearController(self, position)

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
        return self._linear_control.target

    @destination.setter
    def destination(self, position: np.ndarray):
        self._linear_control.target = position

    def update(self, time_step: float):
        linear_acceleration = self._linear_control.derive_acceleration()
        angular_acceleration = np.zeros_like(self.angular_velocity)

        self._linear_motion.update(time_step, linear_acceleration)
        self._angular_motion.update(time_step, angular_acceleration)
