import numpy as np

# Final stage thresholds
POSITION_THRESHOLD = 0.1
LINEAR_VELOCITY_THRESHOLD = 0.1

# Maximum possible accelerations for the ship
LINEAR_ACCELERATION = np.array([2.5, 5.0])


class Controller:
    def __init__(self,
                 parent: "Ship",
                 target_position: np.ndarray):
        self._parent = parent
        self.target_position = target_position

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position

    @target_position.setter
    def target_position(self, position: np.ndarray):
        assert position.shape == self._parent.position.shape

        self._target_position = position

    @staticmethod
    def signed_sqrt(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    def derive_linear_acceleration(self) -> np.ndarray:
        parent_position = self._parent.position
        parent_velocity = self._parent.linear_velocity

        target_position = self._target_position
        target_velocity = np.zeros_like(parent_velocity)

        position_error = target_position - parent_position
        velocity_error = target_velocity - parent_velocity

        final_stage = np.abs(position_error) < POSITION_THRESHOLD
        final_stage &= np.abs(velocity_error) < LINEAR_VELOCITY_THRESHOLD

        ideal_product = position_error * LINEAR_ACCELERATION
        ideal_velocity = self.signed_sqrt(2 * ideal_product)

        ideal_weight = np.sign(ideal_velocity - parent_velocity)
        ideal_acceleration = ideal_weight * LINEAR_ACCELERATION

        final_weight = 0.5 * position_error / POSITION_THRESHOLD
        final_weight += 0.5 * velocity_error / LINEAR_VELOCITY_THRESHOLD
        final_acceleration = final_weight * LINEAR_ACCELERATION

        return np.where(final_stage,
                        final_acceleration,
                        ideal_acceleration)

    def derive_angular_acceleration(self) -> np.ndarray:
        return np.zeros_like(self._parent.angular_velocity)


class Integrator:
    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray):
        assert position.shape == velocity.shape

        self.position = position
        self.velocity = velocity

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

        self._controller = Controller(self, position)

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
        return self._controller.target_position

    @destination.setter
    def destination(self, position: np.ndarray):
        self._controller.target_position = position

    def update(self, time_step: float):
        linear_acceleration = self._controller.derive_linear_acceleration()
        angular_acceleration = self._controller.derive_angular_acceleration()

        self._linear_motion.update(time_step, linear_acceleration)
        self._angular_motion.update(time_step, angular_acceleration)
