# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:04:18 2020

@author: seojin
"""

import numpy as np


def numerical_gradient(f, x):
    x = np.array(x, dtype=np.float64)
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f, init_x, lr, step_num=100):
    x = init_x
    for i in range(0, step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
