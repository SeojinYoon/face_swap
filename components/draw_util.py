# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:38:52 2020

@author: seojin
"""

"""
component about drawing
"""

import matplotlib.pylab as plt
import cv2
from components.face_process import central_axis, triangle_outter_pts
from components.sj_geometry import get_x_interceptPt_with_line, get_image_end_pt_with_line
import numpy as np
from components.landmark_detector import dlib_point2tuple, dlib_points2tuple, detect_partialFaceArea_using_landmark

def draw_landmarks_cv(img, landmarks, radius = 2, thickness = 2, color = (0, 255, 0)):
    """
    represent facial landmarks in image

    :param img: face image
    :param landmarks: landmark data(dlib.point list)
    :param radius: circle radius of landmark
    :param thickness: circle segment size of landmark
    :param color: circle color of landmark
    :return: image represented by facial landmark
    """
    dst = img.copy()

    for pt in landmarks:
        cv2.circle(dst, (pt.x, pt.y), radius = radius, thickness = thickness, color = color)

    return dst

def draw_landmarks(img, landmarks):
    """
    represent facial landmarks in image using matplotlib

    :param img: face image
    :param landmarks: landmark data(dlib.point list)
    """
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    for pt in landmarks:
        circ = plt.Circle( xy = (pt.x, pt.y), radius = 10)
        ax.add_patch(circ)

    plt.show()

def draw_central_axis(img, landmarks, color = (255,0,0), thickness = 10):
    """
    draw face's central axis in image

    :param img: face image
    :param landmarks: landmarks(dlib.point list)
    :param color: rgb
    :param thickness: thickness of straight line
    :return: face image with central axis
    """
    slope, bias = central_axis(landmarks)

    dst = np.zeros(img.shape, dtype=np.uint8)

    line_pt1 = get_x_interceptPt_with_line(slope, bias)
    line_pt2 = get_image_end_pt_with_line(slope, bias, img.shape[0])

    # central axis의 양끝점 검출해서 직선으로 이으면 됨
    cv2.line(dst, line_pt1, line_pt2, color = color, thickness = thickness)

    return dst

def draw_traingles(triangle_pts, img, thickness):
    """
    draw triangle in image

    :param triangle_pts: point list ex) [[[0,0], [1,1], [0,1]]]
    :param img: image
    :param thickness: triangle's line thickness
    :return: image with triangles drawn along points
    """
    dst = img.copy()
    for pts in triangle_pts:    
        cv2.line(dst, tuple(pts[0]), tuple(pts[1]), (255,0,0), thickness)
        cv2.line(dst, tuple(pts[1]), tuple(pts[2]), (255,0,0), thickness)
        cv2.line(dst, tuple(pts[0]), tuple(pts[2]), (255,0,0), thickness)
    return dst

def draw_argument_line(src_img, argument_points, full_connect=False, color=(255, 0, 0), thickness=1):
    """
    draw many line in image

    :param src_img: image
    :param argument_points: points ex) [[1,2], [3,4]]
    :param full_connect: is drawing closed line?
    :param color: color of line
    :param thickness: thickness of line
    :return: image with lines along points
    """
    # image visual processing
    dst = src_img.copy()
    for i, _ in enumerate(argument_points):
        # i: 특징점 Index

        if (i == len(argument_points) - 1):
            # last index에서는 끝점과 첫번째 점을 이어줌
            if full_connect == True:
                cv2.line(dst, argument_points[i], argument_points[0], color=color, thickness=thickness)
        else:
            cv2.line(dst, argument_points[i], argument_points[i + 1], color=color, thickness=thickness)

    return dst


def full_draw(img, landmarks):
    """
    draw face areas in image

    :param img: face image
    :param landmarks: face landmark (dlib.point list)
    :return: image with face areas
    """
    thickness = 1

    if (img.shape[0] * img.shape[1]) > 5171184:
        thickness = 15

    dst = draw_argument_line(src_img=img,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='chin')),
                             thickness=thickness)

    dst = draw_argument_line(src_img=dst,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='left eyebrow')),
                             thickness=thickness)
    dst = draw_argument_line(src_img=dst,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='right eyebrow')),
                             thickness=thickness)

    dst = draw_argument_line(src_img=dst,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='upper nose')),
                             thickness=thickness)

    dst = draw_argument_line(src_img=dst,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='side of nose')),
                             thickness=thickness)

    dst = draw_argument_line(src_img=dst,
                             full_connect=True,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='lips outline')),
                             thickness=thickness)

    dst = draw_argument_line(src_img=dst,
                             argument_points=dlib_points2tuple(detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='lips inline')),
                             thickness=thickness)

    right_cheek_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='right cheek')
    cv2.ellipse(img=dst,
                center=right_cheek_area._center_pt,
                axes=(right_cheek_area._minor_radius, right_cheek_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    left_cheek_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='left cheek')
    cv2.ellipse(img=dst,
                center=left_cheek_area._center_pt,
                axes=(left_cheek_area._minor_radius, left_cheek_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    bellow_left_eye_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='bellow left eye')
    cv2.ellipse(img=dst,
                center=bellow_left_eye_area._center_pt,
                axes=(bellow_left_eye_area._minor_radius, bellow_left_eye_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=180,
                color=(255, 0, 0),
                thickness=thickness)

    left_eye_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='left eye')
    cv2.ellipse(img=dst,
                center=left_eye_area._center_pt,
                axes=(left_eye_area._minor_radius, left_eye_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=180,
                color=(255, 0, 0),
                thickness=thickness)

    right_eye_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='right eye')
    cv2.ellipse(img=dst,
                center=right_eye_area._center_pt,
                axes=(right_eye_area._minor_radius, right_eye_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=180,
                color=(255, 0, 0),
                thickness=thickness)

    bellow_right_eye_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='bellow right eye')
    cv2.ellipse(img=dst,
                center=bellow_right_eye_area._center_pt,
                axes=(bellow_right_eye_area._minor_radius, bellow_right_eye_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=180,
                color=(255, 0, 0),
                thickness=thickness)

    forehead_area = detect_partialFaceArea_using_landmark(landmarks=landmarks,
                                                                                partial_area='forehead')
    cv2.ellipse(img=dst,
                center=forehead_area._center_pt,
                axes=(forehead_area._major_radius, forehead_area._minor_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    nose_area = detect_partialFaceArea_using_landmark(landmarks=landmarks, partial_area='nose')
    cv2.ellipse(img=dst,
                center=nose_area._center_pt,
                axes=(nose_area._minor_radius, nose_area._major_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    side_left_eye = detect_partialFaceArea_using_landmark(landmarks=landmarks,
                                                                                partial_area='side of left eye')
    cv2.ellipse(img=dst,
                center=side_left_eye._center_pt,
                axes=(side_left_eye._minor_radius, side_left_eye._major_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    side_right_eye = detect_partialFaceArea_using_landmark(landmarks=landmarks,
                                                                                 partial_area='side of right eye')
    cv2.ellipse(img=dst,
                center=side_right_eye._center_pt,
                axes=(side_right_eye._minor_radius, side_right_eye._major_radius),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(255, 0, 0),
                thickness=thickness)

    return dst