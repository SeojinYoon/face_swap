# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:36:40 2020

@author: seojin
"""

import numpy as np
import cv2


def shift_2dmask(mask, direction, constant):
    """
    shift mask

    :param mask: mask
    :param direction: direction
    :param constant: padding constant
    :return:
    """
    p_mask = np.pad(mask, (1, 1), mode='constant', constant_values=constant)

    if direction == 'T':
        return p_mask[2:, 1:-1]
    elif direction == 'B':
        return p_mask[0:-2, 1:-1]
    elif direction == 'L':
        return p_mask[1:-1, 2:]
    elif direction == 'R':
        return p_mask[1:-1, 0:-2]


def find_last_nonzero_row(img, is_topfirst):
    """
    find last nonzero row index

    :param img: image
    :param is_topfirst: if the value is true, check from top to bottom
    :return: index
    """
    img_cp = img.copy()

    gray = cv2.cvtColor(img_cp, cv2.COLOR_RGB2GRAY)

    if is_topfirst == True:
        find_nonzero_rowIndex = gray.shape[0] - (gray != 0)[::-1, :].argmax(0)
    else:
        find_nonzero_rowIndex = gray.shape[0] - (gray != 0)[::1, :].argmax(0)

    find_nonzero_rowIndex = np.where(find_nonzero_rowIndex == gray.shape[0], 0, find_nonzero_rowIndex)

    return find_nonzero_rowIndex


def partition_d1(start_value, end_value, partition_count):
    """
    This function make partitions by ranging from start_value to end_value
    if arguments are start_value: 0, end_value: 1, partion_count: 2
        then the result is
        [
            (0, 0.5),
            (0.5, 1)
        ]

    :param start_value: value
    :param end_value: value
    :param partition_count: partion_count
    :return: partition list
    """
    start_x = start_value
    dx = (end_value - start_value) / partition_count

    partitions = []
    for partition_i in range(1, partition_count + 1):
        if partition_i == partition_count:
            partitions.append((start_x, end_value))
        else:
            partitions.append((start_x, start_x + dx))

        start_x += dx
    return partitions


def partition_d1_2(start_value, end_value, partition_count):
    """
    This function make partitions by ranging from start_value to end_value
    if arguments are start_value: 0, end_value: 1, partion_count: 2
        then the result is [0.0, 0.5, 1.0]

    :param start_value: value
    :param end_value: value
    :param partition_count: partion_count
    :return: partition list
    """
    return sorted(list(set(np.array(partition_d1(start_value, end_value, partition_count)).flatten())))


def angle(pivot_vector, other_vector):
    """
    calculate angle between pivot and other

    :param pivot_vector: list ex) [1, 2]
    :param other_vector: list ex) [3, 4]
    :return: radian, + value is clock wise from pivot to other, - value is counter clock wise from pivot to other
    """
    import math

    # https://www.edureka.co/community/32921/signed-angle-between-vectors
    x1, y1 = pivot_vector
    x2, y2 = other_vector

    pivot_angle = math.atan2(y1, x1)
    other_angle = math.atan2(y2, x2)

    return other_angle - pivot_angle


def get_multiple_elements_in_list(in_list, in_indices):
    """
    get multiple values in list

    :param in_list: list
    :param in_indices: indexes
    :return: list of values
    """
    return [in_list[i] for i in in_indices]

