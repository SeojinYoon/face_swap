# https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy

import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def inversion(m):
    minv=np.empty(m.shape,dtype=m.dtype)
    for i in range(m.shape[0]):
        determinant_inv = 1./(m[i,0]*m[i,4]*m[i,8] + m[i,3]*m[i,7]*m[i,2] + m[i,6]*m[i,1]*m[i,5] - m[i,0]*m[i,5]*m[i,7] - m[i,2]*m[i,4]*m[i,6] - m[i,1]*m[i,3]*m[i,8])
        minv[i,0]=(m[i,4]*m[i,8]-m[i,5]*m[i,7])*determinant_inv
        minv[i,1]=(m[i,2]*m[i,7]-m[i,1]*m[i,8])*determinant_inv
        minv[i,2]=(m[i,1]*m[i,5]-m[i,2]*m[i,4])*determinant_inv
        minv[i,3]=(m[i,5]*m[i,6]-m[i,3]*m[i,8])*determinant_inv
        minv[i,4]=(m[i,0]*m[i,8]-m[i,2]*m[i,6])*determinant_inv
        minv[i,5]=(m[i,2]*m[i,3]-m[i,0]*m[i,5])*determinant_inv
        minv[i,6]=(m[i,3]*m[i,7]-m[i,4]*m[i,6])*determinant_inv
        minv[i,7]=(m[i,1]*m[i,6]-m[i,0]*m[i,7])*determinant_inv
        minv[i,8]=(m[i,0]*m[i,4]-m[i,1]*m[i,3])*determinant_inv

    return minv

#I was to lazy to modify the code from the link above more thoroughly
def inversion_3x3(m):
    m_TMP=m.reshape(m.shape[0],9)
    minv=inversion(m_TMP)
    return minv.reshape(minv.shape[0],3,3)

def slow_inverse(A):
    Ainv = np.zeros_like(A)

    for i in range(A.shape[0]):
        Ainv[i] = np.linalg.inv(A[i])
    return Ainv

def matrixIndex_to_Point2D(row, col):
    return (col, row)

def point2D_to_matrixIndex(x, y):
    return (y, x)

