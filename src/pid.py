import datetime
import sys


class PID(object):
    def __init__(self, input, output, setpoint, kp, ki, kd, direct):
        self.output = output
        self.input = input
        self.setpoint = setpoint

        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.disp_kp = 0
        self.disp_ki = 0
        self.disp_kd = 0
        self.i_term = 0
        self.auto = False
        self.direct = None
        self.sample_time = 100
        self.sample_timedelta = 0
        self.output_value = 0
        self.last_input = 0
        self.out_min = -float("inf")
        self.out_max = float("inf")

        self.set_output_limits(0, 255)
        self.sample_time = 100  # milliseconds
        self.direct = direct
        self.set_tunings(kp, ki, kd)

    def initialize(self):
        self.i_term = self.output_value
        self.last_input = self.input
        if self.i_term > self.out_max:
            self.i_term = self.out_max
        elif self.i_term < self.out_min:
            self.i_term = self.out_min

    def compute(self):
        input_value = self.input
        error = self.setpoint - input_value
        self.i_term += self.ki * error
        if self.i_term > self.out_max:
            self.i_term = self.out_max
        elif self.i_term < self.out_min:
            self.i_term = self.out_min

        delta_input = input_value - self.last_input
        output = self.kp * error + self.i_term - self.kd * delta_input

        if output > self.out_max:
            output = self.out_max
        elif output < self.out_min:
            output = self.out_min

        self.output_value = output

        self.last_input = input_value
        return True

    def get_output_value(self, value):
        self.output_value = value
        self.output(value)

    def set_tunings(self, kp, ki, kd):
        if kp < 0 or ki < 0 or kd < 0:
            return

        self.disp_kp = kp
        self.disp_ki = ki
        self.disp_kd = kd

        sample_time_in_sec = self.sample_time / 1000.0
        self.kp = kp
        self.ki = ki * sample_time_in_sec
        self.kd = kd * sample_time_in_sec

        if not self.direct:
            self.kp = 0 - self.kp
            self.ki = 0 - self.ki
            self.kd = 0 - self.kd

    def set_setpoint(self, value):
        self.setpoint = value

    def set_sample_time(self, new_sample_time):
        if new_sample_time <= 0:
            return

        ratio = new_sample_time / float(self.sample_time)
        self.ki *= ratio
        self.kd /= ratio
        self.sample_time = new_sample_time

    def set_output_limits(self, out_min, out_max):
        if out_min >= out_max:
            return
        self.out_min = out_min
        self.out_max = out_max

        if not self.auto:
            return

        if self.output_value > self.out_max:
            self.output_value = self.out_max
        elif self.output_value < self.out_min:
            self.output_value = self.out_min

        if self.i_term > self.out_max:
            self.i_term = self.out_max
        elif self.i_term < self.out_min:
            self.i_term = self.out_min

    def set_auto(self, new_auto):
        if new_auto != self.auto:
            self.initialize()
        self.auto = new_auto

    def set_direction(self, value):
        if self.auto and value != self.direct:
            self.kp = 0 - self.kp
            self.ki = 0 - self.ki
            self.kd = 0 - self.kd
        self.direct = value

    def get_kp(self):
        return self.disp_kp

    def get_ki(self):
        return self.disp_ki

    def get_kd(self):
        return self.disp_kd
