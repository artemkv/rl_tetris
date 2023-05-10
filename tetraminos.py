import numpy as np

o_shape = [np.array([[1, 1],
                     [1, 1]]),
           np.array([[1, 1],
                     [1, 1]]),
           np.array([[1, 1],
                     [1, 1]]),
           np.array([[1, 1],
                     [1, 1]])]

s_shape = [np.array([[0, 1, 1],
                     [1, 1, 0],
                     [0, 0, 0]]),
           np.array([[0, 1, 0],
                     [0, 1, 1],
                     [0, 0, 1]]),
           np.array([[0, 0, 0],
                     [0, 1, 1],
                     [1, 1, 0]]),
           np.array([[1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0]])]

z_shape = [np.array([[1, 1, 0],
                     [0, 1, 1],
                     [0, 0, 0]]),
           np.array([[0, 0, 1],
                     [0, 1, 1],
                     [0, 1, 0]]),
           np.array([[0, 0, 0],
                     [1, 1, 0],
                     [0, 1, 1]]),
           np.array([[0, 1, 0],
                     [1, 1, 0],
                     [1, 0, 0]])]

j_shape = [np.array([[0, 1, 0],
                     [0, 1, 0],
                     [1, 1, 0]]),
           np.array([[1, 0, 0],
                     [1, 1, 1],
                     [0, 0, 0]]),
           np.array([[0, 1, 1],
                     [0, 1, 0],
                     [0, 1, 0]]),
           np.array([[0, 0, 0],
                     [1, 1, 1],
                     [0, 0, 1]])]

l_shape = [np.array([[0, 1, 0],
                     [0, 1, 0],
                     [0, 1, 1]]),
           np.array([[0, 0, 0],
                     [1, 1, 1],
                     [1, 0, 0]]),
           np.array([[1, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]]),
           np.array([[0, 0, 1],
                     [1, 1, 1],
                     [0, 0, 0]])]

t_shape = [np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 0, 0]]),
           np.array([[0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 0]]),
           np.array([[0, 0, 0],
                     [1, 1, 1],
                     [0, 1, 0]]),
           np.array([[0, 1, 0],
                     [1, 1, 0],
                     [0, 1, 0]])]

i_shape = [np.array([[0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0]]),
           np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]),
           np.array([[0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0]]),
           np.array([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])]

# tetraminos = [o_shape, s_shape, z_shape, j_shape, l_shape, t_shape, i_shape];
tetraminos = [o_shape, t_shape, l_shape, s_shape, z_shape, j_shape, i_shape]
