from __future__ import division, print_function, absolute_import

from MackeyGlass import MackeyGlass as MG
import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')   # use('Agg') for saving to file and use('TkAgg') for interactive plot
import matplotlib.pyplot as plt


mg = MG()
timeSeries = mg.generate_samples(300)

print(np.shape(timeSeries))

plt.figure(figsize=(16, 4))
plt.suptitle('Mackey-Glass Time Series')
plt.plot(timeSeries, 'g-', label='Output')
plt.legend()
plt.show()