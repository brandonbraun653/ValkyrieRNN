import numpy as np


class PID(object):
    DIRECT = False
    REVERSE = True

    def __init__(self, input, output, setpoint, kp, ki, kd, direct):
        """

        :param input: A function that returns an input value
        :param output: A function that sets an output value
        :param setpoint: A scalar for an initial setpoint
        :param kp: Scalar
        :param ki: Scalar
        :param kd: Scalar
        :param direct: Boolean Describes action direction of PID
        """
        # Some input checking to prevent errors later
        assert(callable(input))
        assert(callable(output))
        assert(np.isscalar(setpoint))
        assert(np.isscalar(kp))
        assert(np.isscalar(ki))
        assert(np.isscalar(kd))

        self._output = output
        self._input = input
        self._setpoint = setpoint

        self._kp = 0
        self._ki = 0
        self._kd = 0
        self._disp_kp = 0
        self._disp_ki = 0
        self._disp_kd = 0
        self._output_sum = None
        self._auto = False
        self._direct = None
        self._sample_time = 100
        self._sample_timedelta = None
        self._output_value = 0
        self._last_input = None
        self._out_min = -float('inf')
        self._out_max = float('inf')

        self.set_output_limits(0, 255)
        self.sample_time = 100  # milliseconds
        self.controller_direction = direct
        self.set_tunings(kp, ki, kd)

    def initialize(self):
        self._output_sum = self.output_value
        self._last_input = self._input()

        if self._output_sum > self._out_max:
            self._output_sum = self._out_max
        elif self._output_sum < self._out_min:
            self._output_sum = self._out_min

    def compute(self):
        input_value = self._input()
        error = (self.setpoint - input_value)
        dInput = (input_value - self._last_input)
        self._output_sum += self._ki * error

        if self._output_sum > self._out_max:
            self._output_sum = self._out_max
        elif self._output_sum < self._out_min:
            self._output_sum = self._out_min

        self.output_value = self._kp * error

        if self.output_value > self._out_max:
            self.output_value = self._out_max
        elif self.output_value < self._out_min:
            self.output_value = self._out_min

        self.output_value += self._output_sum - (self._kd * dInput)

        self._last_input = input_value
        return True

    @property
    def output_value(self):
        return self._output_value

    @output_value.setter
    def output_value(self, value):
        self._output_value = value
        self._output(value)

    def set_tunings(self, kp, ki, kd):
        if kp < 0 or ki < 0 or kd < 0:
            return

        self._disp_kp = kp
        self._disp_ki = ki
        self._disp_kd = kd

        sample_time_in_sec = self.sample_time / 1000.0
        self._kp = kp
        self._ki = ki * sample_time_in_sec
        self._kd = kd / sample_time_in_sec

        if self.controller_direction == self.REVERSE:
            self._kp = 0 - self._kp
            self._ki = 0 - self._ki
            self._kd = 0 - self._kd

    @property
    def setpoint(self):
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value):
        self._setpoint = value

    @property
    def sample_time(self):
        return self._sample_time

    @sample_time.setter
    def sample_time(self, new_sample_time):
        if new_sample_time > 0:
            ratio = float(new_sample_time) / float(self._sample_time)
            self._ki *= ratio
            self._kd /= ratio
            self._sample_time = new_sample_time

    def set_output_limits(self, out_min, out_max):
        if out_min >= out_max:
            return

        self._out_min = out_min
        self._out_max = out_max

        if self.auto:
            if self.output_value > self._out_max:
                self.output_value = self._out_max
            elif self.output_value < self._out_min:
                self.output_value = self._out_min

            if self._output_sum > self._out_max:
                self._output_sum = self._out_max
            elif self._output_sum < self._out_min:
                self._output_sum = self._out_min

    @property
    def auto(self):
        return self._auto

    @auto.setter
    def auto(self, new_auto):
        if new_auto != self._auto:
            self.initialize()
        self._auto = new_auto

    @property
    def controller_direction(self):
        return self._direct

    @controller_direction.setter
    def controller_direction(self, value):
        if self.auto and value != self._direct:
            self._kp = 0 - self._kp
            self._ki = 0 - self._ki
            self._kd = 0 - self._kd
        self._direct = value

    @property
    def kp(self):
        return self._disp_kp

    @property
    def ki(self):
        return self._disp_ki

    @property
    def kd(self):
        return self._disp_kd
