import matlab.engine
import numpy as np
import time
from Scripts.Matlab.MatlabIOHelper import numpy_to_matlab_matrix, matlab_matrix_to_numpy

if __name__ == '__main__':
    # -----------------------------
    # Setup the Matlab Engine
    # -----------------------------
    print("Starting Matlab Engine")
    eng = matlab.engine.start_matlab()
    eng.addpath(r'C:\git\GitHub\ValkyrieRNN\Scripts\Matlab', nargout=0)
    print("Done")

    x = np.zeros([1, 1000])
    y = np.zeros([1, 1000])

    start_time = time.perf_counter()
    mX = numpy_to_matlab_matrix(x)
    mY = numpy_to_matlab_matrix(y)

    stepData = eng.CalculateStepPerformance(mX, mY, nargout=1)
    end_time = time.perf_counter()

    print(stepData['RiseTime'])
    print(end_time-start_time)
