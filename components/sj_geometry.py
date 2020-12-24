# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:55:49 2020

@author: seojin
"""

import numpy as np


def proj_pt2line(pt, direction_vector, x_intercept):
    """
    project poting to line

    :param pt: list(point) ex [1,1]
    :param direction_vector: list(vector) ex np.array([1,1])
    :param x_intercept: to specify located vector ex) 3
    :return: list(projection point)
    """
    w = pt

    cv = (np.dot(w, direction_vector)) / np.dot(direction_vector, direction_vector) * np.array(direction_vector)

    return [cv[0] + x_intercept, cv[1]]


def get_warp_affine_transformation_with_list(origin_rs, warped_rs):
    return get_warp_affine_transformation(origin_rs[0], origin_rs[1], origin_rs[2], warped_rs[0], warped_rs[1],
                                          warped_rs[2])


# input: pt
def get_warp_affine_transformation(origin_r_i, origin_r_j, origin_r_k, warped_r_i, warped_r_j, warped_r_k):
    """
    calculate affine transformation matrix, origin triangle to warped triangle

    :param origin_r_i: 1st orgin triangle position ex) [1,1]
    :param origin_r_j: 2st orgin triangle position ex) [1,2]
    :param origin_r_k: 3st orgin triangle position ex) [3,1]
    :param warped_r_i: 1st warped triangle position ex) [1,1]
    :param warped_r_j: 2st warped triangle position ex) [1,2]
    :param warped_r_k: 3st warped triangle position ex) [4,1]
    :return: Affine transformation Matrix
    """
    # R = M * R^
    # return M
    R_warp = np.array([[warped_r_i[0], warped_r_j[0], warped_r_k[0]], [warped_r_i[1], warped_r_j[1], warped_r_k[1]]])
    R_hat = np.array(
        [[origin_r_i[0], origin_r_j[0], origin_r_k[0]], [origin_r_i[1], origin_r_j[1], origin_r_k[1]], [1, 1, 1]])

    return np.dot(R_warp, np.linalg.inv(R_hat))


def get_warp_affine_transformation_polygon(origin_rs, warped_rs):
    """
    calculate affine transformation matrix, origin_rs to warped_rs

    :param origin_rs: orgin positions ex) [[1,1] ...]
    :param warped_rs: warped positions ex) [[1,1] ...]
    :return: Affine transformation Matrix
    """
    # R = M * R^
    # return M

    R_warp = np.array([[wp_pt[0] for wp_pt in warped_rs], [wp_pt[1] for wp_pt in warped_rs]])
    R_hat = np.array(
        [[origin_pt[0] for origin_pt in origin_rs], [origin_pt[1] for origin_pt in origin_rs], [1, 1, 1, 1]])

    return np.dot(R_warp, np.linalg.inv(R_hat))


def dist_lineWithPt(pt, slope, bias):
    """
    find distance between pt and line

    :param pt: list(point) ex) [1,2]
    :param slope: slope ex) 3
    :param bias: bias ex) 4
    :return: distance
    """
    x = pt[0]
    y = pt[1]

    a, b, c = line_normal2standard(slope, bias)

    # canonical form
    return abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)


def dist_pt2pt(pt1, pt2):
    """
    find distance between pt1 and pt2

    :param pt1: list(pt) ex) [1,2]
    :param pt2: list(pt) ex) [3,4]
    :return: distnace
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def line_normal2standard(slope, bias):
    """
    line is converted normal form to canonical form

    :param slope: slope ex) 4
    :param bias: bias ex) 2
    :return:
    """
    a = -slope
    b = 1
    c = -bias

    return (a, b, c)


def opposite_pt2line(pt, slope, bias):
    """
    point is applied opposite transition at line

    :param pt: list(pt) ex) [1,2]
    :param slope: slope ex) 3
    :param bias: bias ex) 4
    :return: list(point)
    """
    x = pt[0]
    y = pt[1]

    a, b, c = line_normal2standard(slope, bias)

    x_ = x - ((2 * a) * (a * x + b * y + c)) / (a ** 2 + b ** 2)
    y_ = y - ((2 * b) * (a * x + b * y + c)) / (a ** 2 + b ** 2)
    return (x_, y_)


def get_x_interceptPt_with_line(slope, bias):
    """
    find x intercept in line

    :param slope: slope
    :param bias: bias
    :return: x intercept
    """
    return (int((-bias) / slope), 0)


# 이미지의 해상도의 끝점에 대한 line의 end point를 찾음
def get_image_end_pt_with_line(slope, bias, image_end_y):
    """
    find end point of line at end point of image resolution

    :param slope: line's slope
    :param bias: line's bias
    :param image_end_y: y position about image resolution
    :return: line's end point
    """
    return (int((image_end_y - bias) / slope), int(image_end_y))


class SJ_ellipse:
    """
    Ellipse

    Property:
    center_pt ex) [1,2]
    minor_radius ex) 3
    major_radius ex) 4
    """

    def __init__(self, center_pt, minor_radius, major_radius):
        self._center_pt = (center_pt[0], center_pt[1])
        self._minor_radius = minor_radius
        self._major_radius = major_radius
