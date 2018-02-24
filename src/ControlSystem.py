
from src.pid import PID


class AxisController:
    """
    Models the control system found on the ValkyrieFCS flight controller. This
    consists of two PID loops, an inner loop controlling angular rates and an outer
    loop controlling attitude angle.
    """
    def __init__(self, angle_setpoint, angle_fb, rate_fb, output, sample_time_ms):
        """
        Sets up the primary controller structure and initializes the states

        :param angle_setpoint: Desired value for controller to achieve
        :param angle_fb: Angle state feedback signal
        :param rate_fb: Angular rate state feedback signal
        :param output: Output motor control signal
        """
        self._angle_setpoint = angle_setpoint
        self._rate_setpoint = 0     # Currently zero as we are only testing angle control modes

        self._rate_kp = 0
        self._rate_ki = 0
        self._rate_kd = 0
        self._angle_kp = 0
        self._angle_ki = 0
        self._angle_kd = 0

        self._angle_feedback = angle_fb
        self._rate_feedback = rate_fb
        self._angular_rate_desired = 0
        self._motor_cmd_output = output

        # True == DIRECT
        # False == REVERSE
        self._angle_controller_direction = True
        self._rate_controller_direction = False

        self._max_angular_rate = 100.0
        self._motor_output_range = 500.0

        self.angleController = PID(self._angle_feedback,            # Feedback from AHRS of actual angle
                                   self._angular_rate_desired,      # Output into the rateController
                                   self._angle_setpoint,            # Set by user as angle to achieve
                                   self._angle_kp, self._angle_ki, self._angle_kd,
                                   self._angle_controller_direction)

        self.angleController.auto(False)
        self.angleController.sample_time(sample_time_ms)
        self.angleController.set_output_limits(-self._max_angular_rate, self._max_angular_rate)

        self.rateController = PID(self._rate_feedback,              # Feedback from GYRO of actual rate
                                  self._motor_cmd_output,           # Output into the NN
                                  self._angular_rate_desired,       # Output from the angleController
                                  self._rate_kp, self._rate_ki, self._rate_kd,
                                  self._rate_controller_direction)

        self.rateController.auto(False)
        self.rateController.sample_time(sample_time_ms)
        self.rateController.set_output_limits(-self._motor_output_range, self._motor_output_range)

    def update_angle_pid(self, kp, ki, kd):
        self._angle_kp = kp
        self._angle_ki = ki
        self._angle_kd = kd
        self.angleController.set_tunings(kp, ki, kd)

    def update_rate_pid(self, kp, ki, kd):
        self._rate_kp = kp
        self._rate_ki = ki
        self._rate_kd = kd
        self.rateController.set_tunings(kp, ki, kd)

    def compute(self):
        """
        Computes a new motor output signal assuming the variables passed to
        the class in the __init__ function have been updated appropriately
        """
        self.angleController.compute()
        self.rateController.compute()


if __name__ == "__main__":
    print('hello')