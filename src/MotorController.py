import numpy as np

motor_mix_matrix = np.array([[1.0,  1.0,  1.0, 0.0],
                            [1.0, -1.0, -1.0, 0.0],
                            [1.0,  1.0, -1.0, 0.0],
                            [1.0, -1.0,  1.0, 0.0]])
max_motor_val = 1650
min_motor_val = 1060


def limit_motor_signal(sig):
    if sig > max_motor_val:
        sig = max_motor_val
    elif sig < min_motor_val:
        sig = min_motor_val
    return sig


def generate_motor_signals(pid_throttle, pid_pitch, pid_roll, pid_yaw):
    sig = np.matmul(motor_mix_matrix, np.array([[pid_throttle], [pid_pitch], [pid_roll], [pid_yaw]]))

    sig[0, 0] = limit_motor_signal(sig[0, 0])
    sig[1, 0] = limit_motor_signal(sig[1, 0])
    sig[2, 0] = limit_motor_signal(sig[2, 0])
    sig[3, 0] = limit_motor_signal(sig[3, 0])

    return sig


def process_raw_motor_signals_for_rnn(motor_signal):
    # Do some things here to scale from the huge output to a more sensible range
    raise NotImplementedError


if __name__ == "__main__":
    throttle = 1250
    pitch = -100
    roll = 250
    yaw = 0

    motor_cmd_raw = generate_motor_signals(throttle, pitch, roll, yaw)
    print(motor_cmd_raw)
