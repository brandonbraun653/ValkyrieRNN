import matlab.engine
import numpy as np


def numpy_matrix_to_matlab(np_mat, data_type='double'):
    """
    Converts any numpy array into a variable that can be passed to Matlab. Full
    support of data types can be found here:
    https://www.mathworks.com/help/matlab/matlab_external/handle-data-returned-from-matlab-to-python.html

    :param np_mat: Input numpy.ndarray type
    :param data_type: String of data type desired
    :return: Matlab matrix equivalent of input numpy data
    """
    if data_type == 'double':
        return matlab.double(np_mat.tolist())

    elif data_type == 'single':
        return matlab.single(np_mat.tolist())

    elif data_type == 'uint8':
        return matlab.uint8(np_mat.tolist())

    elif data_type == 'uint16':
        return matlab.uint16(np_mat.tolist())

    else:
        raise ValueError("Data type not supported. Add it.")


def matlab_matrix_to_numpy(matlab_mat):
    """
    Converts a Matlab matrix into a numpy ndarrary with arbitrary dimensions
    :param matlab_mat: Matlab matrix
    :return: Numpy ndarray
    """
    return np.array(matlab_mat._data).reshape(matlab_mat.size, order='F')
