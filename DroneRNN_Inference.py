from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as graph



inputs = [np.arange(5), np.arange(5), np.arange(5), np.arange(5)]
inputs = np.reshape(inputs, (4, 5))
print(inputs)

inputs = np.roll(inputs, 1, axis=1)
print(inputs)

for i in range(5, 10):
    print(i)
    inputs[:,0] = np.array([i, i, i, i])
    print(inputs)

    print('Rolling...')
    inputs = np.roll(inputs, 1, axis=1)
    print(inputs)
    print('\n')

if __name__ == "__main__":
    print("Yo fam")