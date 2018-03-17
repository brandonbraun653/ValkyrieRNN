
from src.PIDController import PID


class PIDInputs:
    def __init__(self):
        self._angle_fb = -1234

    @property
    def angle_fb(self):
        return self._angle_fb


class AxisController:
    """
    Models the control system found on the ValkyrieFCS flight controller. This
    consists of two PID loops, an inner loop controlling angular rates and an outer
    loop controlling attitude angle.
    """
    def __init__(self, angular_rate_range, motor_cmd_range, angle_direction, rate_direction, sample_time_ms):
        self._angle_setpoint = 0
        self._rate_setpoint = 0

        self._angle_feedback = 0
        self._rate_feedback = 0
        self._angular_rate_desired = 0
        self._motor_cmd_output = 0
        self._sample_time_ms = sample_time_ms

        # Setup the Angle Control PID Object
        self.angleController = PID(lambda: self.angle_feedback,         # Feedback from AHRS of actual angle
                                   self._update_angular_rate_desired,   # Output into the rateController
                                   self._angle_setpoint,                # Initial condition
                                   0.0,
                                   0.0,
                                   0.0,
                                   angle_direction)

        self.angleController.auto = False
        self.angleController.sample_time = self._sample_time_ms
        self.angleController.set_output_limits(-angular_rate_range, angular_rate_range)
        self.angleController.initialize()

        # Set up the Rate Control PID Object
        self.rateController = PID(lambda: self.rate_feedback,           # Feedback from GYRO of actual rate
                                  self._update_controller_output,       # Output into the NN
                                  self._angular_rate_desired,           # Initial Condition
                                  0.0,
                                  0.0,
                                  0.0,
                                  rate_direction)

        self.rateController.auto = False
        self.rateController.sample_time = self._sample_time_ms
        self.rateController.set_output_limits(-motor_cmd_range, motor_cmd_range)
        self.rateController.initialize()

    @property
    def angle_setpoint(self):
        return self._angle_setpoint

    @angle_setpoint.setter
    def angle_setpoint(self, value):
        self._angle_setpoint = value

    @property
    def angle_feedback(self):
        return self._angle_feedback

    @angle_feedback.setter
    def angle_feedback(self, value):
        self._angle_feedback = value

    @property
    def rate_feedback(self):
        return self._rate_feedback

    @rate_feedback.setter
    def rate_feedback(self, value):
        self._rate_feedback = value

    @property
    def angular_rate_desired(self):
        return self._angular_rate_desired

    @property
    def controller_output(self):
        return self._motor_cmd_output

    def _update_angular_rate_desired(self, value):
        self._angular_rate_desired = value

    def _update_controller_output(self, value):
        self._motor_cmd_output = value

    def update_angle_pid(self, kp, ki, kd):
        self.angleController.set_tunings(kp, ki, kd)

    def update_rate_pid(self, kp, ki, kd):
        self.rateController.set_tunings(kp, ki, kd)

    def compute(self, use_rate_control=False):
        self.angleController.setpoint = self.angle_setpoint
        self.angleController.compute()

        if use_rate_control:
            self.rateController.setpoint = self.angular_rate_desired
            self.rateController.compute()
        else:
            self._update_controller_output(self.angular_rate_desired)



if __name__ == "__main__":
    print('hello')
