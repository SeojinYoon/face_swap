# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:51:30 2020

@author: seojin
"""

"""
component about face 
"""

import dlib
import numpy as np
from components.landmark_detector import dlib_landmark_range, dlib_detection_parser, ref_landmark_warping, \
    dlib_landmark_faceArea, dlib_indexes
from components.sj_geometry import proj_pt2line, get_warp_affine_transformation_with_list, get_x_interceptPt_with_line
from components.sj_util import angle, get_multiple_elements_in_list, partition_d1
from components.sj_image_process import shift_image, remove_overlapped_pixels, Image_preprocessing, find_obj_rectangle
from components.sj_keras_toolbox import find_faceRoi_withParser
import scipy.spatial
import cv2


def central_axis(landmarks):
    """
    calculate slope and bias about central axis of face

    :param landmarks: landmarks(dlib point list)
    :return: (slope, bias)
    """
    upper_nose_landmarks = dlib_landmark_faceArea(landmarks, 'upper nose')

    first_pt = upper_nose_landmarks[0]
    end_pt = upper_nose_landmarks[-1]

    s_denominator = first_pt.x - end_pt.x
    s_numerator = first_pt.y - end_pt.y

    if s_denominator == 0:
        s_denominator = 0.000001

    # y - y1 = (y2 - y1) / (x2-x1)(x - x1)
    # 논문에서는 fitting을 하는데, 여기선 fitting이 잘안되므로 그냥 콧대를 중심축으로 사용

    slope = s_numerator / s_denominator

    bias = (-slope) * end_pt.x + end_pt.y

    return (slope, bias)


def central_axis_vectorForm(landmarks):
    """
    calculate vector about central axis of face

    :param landmarks: landmarks(dlib point list)
    :return: calculate central axis's vector
    """
    m, b = central_axis(landmarks)

    vector = np.array([1, m])

    bias_vector = (0, b)

    return (vector, bias_vector)


def diff_eye_angle(I_in_landmarks, I_ref_landmarks):
    """
    calculate difference between eye angle of input image and eye angle of reference image

    :param I_in_landmarks: input image's landmark
    :param I_ref_landmarks: reference image's landmark
    :return: difference between two eye's angle
    """
    return eye_angle(I_ref_landmarks) - eye_angle(I_in_landmarks)


def eye_angle(landmarks):
    """
    calculate eye angle (find eye angle using center position of left eye and right eye
`   calculate arctan(dY / dX) and make the result to degree

    :param landmarks: landmarks(dlib point list)
    :return: eye's angle(degree)
    """
    # https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = dlib_detection_parser()['left eye']
    (rStart, rEnd) = dlib_detection_parser()['right eye']

    leftEyePts = np.array(landmarks[lStart: lEnd + 1])
    rightEyePts = np.array(landmarks[rStart: rEnd + 1])

    leftEyeCenter = leftEyePts.mean(axis=0)
    rightEyeCenter = rightEyePts.mean(axis=0)

    leftEyeCenter = (int(leftEyeCenter.x), int(leftEyeCenter.y))
    rightEyeCenter = (int(rightEyeCenter.x), int(rightEyeCenter.y))

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    return angle


def find_center_pt(landmarks):
    """
    find landmark's center position
    - project every landmark to central axis and calculate mean of projection datas

    :param landmarks: landmarks (dlib point list)
    :return: landmark's center position
    """
    slope, bias = central_axis(landmarks)
    d_v, b = central_axis_vectorForm(landmarks)

    x_intercept = get_x_interceptPt_with_line(slope, bias)[0]

    projs = []
    for landmark in landmarks:
        origin = [landmark.x, landmark.y]

        pj = proj_pt2line(origin, d_v, x_intercept)
        projs.append(pj)
    return np.mean(np.array(projs), axis=0)


def average_distance(landmarks, landmark_center_pt):
    """
    average distance between each landmark and center point of landmarks
    -

    :param landmarks: landmark(dlib point list)
    :param landmark_center_pt: center point
    :return: average distance
    """
    result = 0
    for landmark in landmarks:
        pt = np.array([landmark.x, landmark.y])
        diff = pt - np.array(landmark_center_pt)

        result += np.linalg.norm(diff)

    result = result / len(landmarks)
    return result


def triangle_outter_pts(landmarks, img_height, img_width):
    """
    triangle's point about image without face area

    :param landmarks: landmarks(dlib point list)
    :param img_height: height of image
    :param img_width: width of image
    :return: triangle's point list
    """
    left_top_corner = [0, 0]
    right_top_corner = [img_width - 1, 0]
    left_bottom_corner = [0, img_height - 1]
    right_bottom_corner = [img_width - 1, img_height - 1]

    dlib_parser = dlib_detection_parser()
    chin_landmark_indxes = dlib_parser['chin']
    chin_start_index, chin_end_index = chin_landmark_indxes[0], chin_landmark_indxes[-1]

    chin_center = int((chin_start_index + chin_end_index) / 2)

    triangle_pts = []
    for i in range(0, chin_center + 1):
        if i == 0:
            triangle_pts.append([left_top_corner, [landmarks[i].x, landmarks[i].y], left_bottom_corner])
        else:
            triangle_pts.append(
                [[landmarks[i - 1].x, landmarks[i - 1].y], [landmarks[i].x, landmarks[i].y], left_bottom_corner])

    for i in range(chin_end_index, chin_center - 1, -1):
        if i == chin_end_index:
            triangle_pts.append([right_top_corner, [landmarks[i].x, landmarks[i].y], right_bottom_corner])
        else:
            triangle_pts.append(
                [[landmarks[i + 1].x, landmarks[i + 1].y], [landmarks[i].x, landmarks[i].y], right_bottom_corner])

    triangle_pts.append([left_bottom_corner, [landmarks[chin_center].x, landmarks[chin_center].y], right_bottom_corner])

    return triangle_pts


def img_warp_bottom_to_face(I_ref, I_ref_landmarks, I_ref_warped_landmarks):
    """
    warping image from bottom to face

    :param I_ref: reference image
    :param I_ref_landmarks: reference image's landmark (dlib point list)
    :param I_ref_warped_landmarks: reference image's warped landmark (dlib point list)
    :return: image which processed warping (bottom to face)
    """
    face_indexs = [i for i in range(dlib_landmark_range()[0], dlib_landmark_range()[1] + 1)]

    img_src_h, img_src_w = I_ref.shape[0], I_ref.shape[1]

    list_I_ref_landmarks = np.array(
        list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))
    list_I_ref_warped_landmarks = np.array(
        list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_warped_landmarks, face_indexs))))

    # 들로네 삼각 분할을 적용하여 얼굴 영역 분할
    triangles = scipy.spatial.Delaunay(
        list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))

    delaunay_trianglePt_indexes = triangles.simplices

    I_ref_triangles = list_I_ref_landmarks[delaunay_trianglePt_indexes].tolist()
    I_warp_triangles = list_I_ref_warped_landmarks[delaunay_trianglePt_indexes].tolist()

    # 얼굴이외의 배경 영역을 warping하기 위하여 삼각형 추가
    I_ref_triangles += triangle_outter_pts(I_ref_landmarks, img_src_h, img_src_w)
    I_warp_triangles += triangle_outter_pts(I_ref_warped_landmarks, img_src_h, img_src_w)

    results = []
    for i in range(0, len(I_ref_triangles)):
        ref_triangle_pt = I_ref_triangles[i]
        warped_ref_triangle_pt = I_warp_triangles[i]

        res = warping_triangle(ref_triangle_pt, warped_ref_triangle_pt, I_ref)
        results.append(res)

    I_ref_warp = np.zeros((img_src_h, img_src_w, 3), np.uint8)
    for warp_triangle in results:
        # warping된 삼각형에서 이미 픽셀이 채워진 영역은 필요없으므로 삭제
        warped_triangle = remove_overlapped_pixels(warp_triangle, I_ref_warp)
        I_ref_warp = cv2.add(I_ref_warp, warped_triangle)

    return I_ref_warp


def warping_triangle(ref_triangle_pt, warped_ref_triangle_pt, img):
    """
    crop image along triangle points and warp the image

    :param ref_triangle_pt: reference image's triangle points ex) [[0,0], [1,1], [0,1]]
    :param warped_ref_triangle_pt: warped triangle points ex) [[0,1], [1,2], [0,3]]
    :param img: image
    :return: warped image(cropped by triangle)
    """
    height, width = img.shape[0], img.shape[1]

    cropped_tr_mask = np.zeros((height, width), np.uint8)

    cv2.fillConvexPoly(cropped_tr_mask, np.array(ref_triangle_pt), 255)
    cropped_triangle = cv2.bitwise_and(img, img, mask=cropped_tr_mask)

    # 원본 영상의 얼굴 삼각형에 대해 warping 된 얼굴 삼각형간의 관계를 파악
    M = get_warp_affine_transformation_with_list(ref_triangle_pt, warped_ref_triangle_pt)

    # 파악한 관계를 가지고 warping된 얼굴 삼각형 생성
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (width, height), flags=cv2.WARP_FILL_OUTLIERS)

    return warped_triangle


def img_warp_top_to_face(I_ref, I_ref_landmarks, I_ref_warped_landmarks):
    """
    warping image from top to face

    :param I_ref: reference image
    :param I_ref_landmarks: reference image's landmark (dlib point list)
    :param I_ref_warped_landmarks: reference image's warped landmark (dlib point list)
    :return: image which processed warping (top to face)
    """
    face_indexs = [i for i in range(dlib_landmark_range()[0], dlib_landmark_range()[1] + 1)]

    img_src_h, img_src_w = I_ref.shape[0], I_ref.shape[1]

    chin_indexes = dlib_indexes('chin')
    upper_face_indexes = [chin_indexes[0]] + dlib_indexes('left eyebrow') + dlib_indexes('right eyebrow') + [
        chin_indexes[-1]]

    # upper_face_indexes = [0] + [17,18,19,20,21,22,23,24,25,26] + [16]

    list_I_ref_landmarks = np.array(
        list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_landmarks, face_indexs))))
    list_I_ref_warped_landmarks = np.array(
        list(map(lambda x: [x.x, x.y], get_multiple_elements_in_list(I_ref_warped_landmarks, face_indexs))))

    # 이미지의 맨위에 있는 line을 분할
    partition_pts = list(map(lambda x: [x, 0], [int(pt[0]) for pt in partition_d1(0, img_src_w, 10)] + [img_src_w - 1]))

    # 눈 주위에 있는 필요한 landmark 뽑아내기
    I_ref_eyebrow_landmarks = get_multiple_elements_in_list(list_I_ref_landmarks.tolist(), upper_face_indexes)
    I_ref_warp_eyebrow_landmarks = get_multiple_elements_in_list(list_I_ref_warped_landmarks.tolist(),
                                                                 upper_face_indexes)

    # 분할된 점과 landmark를 통합
    I_ref_background_pts = I_ref_eyebrow_landmarks + partition_pts
    I_warped_ref_background_pts = I_ref_warp_eyebrow_landmarks + partition_pts

    # reference image, warped_reference image 분할
    triangles = scipy.spatial.Delaunay(I_ref_background_pts)
    delaunay_background_indexes = triangles.simplices

    delaunay_index_map = {
        'first': 0,
        'last': 11
    }
    delaunay_background_indexes = np.array(
        list(filter(lambda x: sum(np.isin(x, range(delaunay_index_map['first'], delaunay_index_map['last'] + 1))) != 3,
                    delaunay_background_indexes)))

    I_ref_background_triangle_pts = np.array(I_ref_background_pts)[delaunay_background_indexes]
    I_warped_ref_background_triangle_pts = np.array(I_warped_ref_background_pts)[delaunay_background_indexes]

    # Warping Process
    results = []
    for i in range(0, len(I_ref_background_triangle_pts)):
        ref_triangle_pt = I_ref_background_triangle_pts[i]
        warped_ref_triangle_pt = I_warped_ref_background_triangle_pts[i]

        res = warping_triangle(ref_triangle_pt, warped_ref_triangle_pt, I_ref)
        results.append(res)

    I_ref_cover = np.zeros((img_src_h, img_src_w, 3), np.uint8)
    for warp_triangle in results:
        # warping된 삼각형에서 이미 픽셀이 채워진 영역은 필요없으므로 삭제
        warped_triangle = remove_overlapped_pixels(warp_triangle, I_ref_cover)
        I_ref_cover = cv2.add(I_ref_cover, warped_triangle)

    return I_ref_cover


def main_face_align(I_in, I_ref, I_in_landmarks, I_ref_landmarks, detector, predictor):
    """
    align image

    :param I_in: input image
    :param I_ref: reference image
    :param I_in_landmarks: input image's landmark
    :param I_ref_landmarks: reference image's landmark
    :param detector: dlib detector
    :param predictor: dlib predictor
    :return: image and landmarks which processed align task
    """
    I_in_c_s = find_center_pt(I_in_landmarks)
    I_ref_c_s = find_center_pt(I_ref_landmarks)

    desired_face_measure = average_distance(I_ref_landmarks, I_ref_c_s)
    input_face_measure = average_distance(I_in_landmarks, I_in_c_s)
    k = desired_face_measure / input_face_measure

    rot_angle = diff_eye_angle(I_in_landmarks, I_ref_landmarks)

    # image scaling, face rotating
    print("diff angle 2: ", rot_angle)
    print("scale value is: ", k)

    if rot_angle > 0:
        I_in_rot = Image_preprocessing.rotate_center(I_in, np.abs(rot_angle), I_in_c_s, True, k)
    else:
        I_in_rot = Image_preprocessing.rotate_center(I_in, np.abs(rot_angle), I_in_c_s, False, k)

    I_in_align = I_in_rot

    I_in_align_landmarks = Image_preprocessing.rotate_point(list(map(lambda x: [x.x, x.y], I_in_landmarks)), I_in_c_s,
                                                            rot_angle, k)
    I_in_align_landmarks = list(map(lambda x: dlib.point(x[0], x[1]), I_in_align_landmarks))

    I_ref_warped_landmarks = ref_landmark_warping(I_in_align_landmarks, I_ref_landmarks, I_in_c_s, I_ref_c_s)

    warped_bottom = img_warp_bottom_to_face(I_ref, I_ref_landmarks, I_ref_warped_landmarks)
    warped_cover = img_warp_top_to_face(I_ref, I_ref_landmarks, I_ref_warped_landmarks)
    warped_cover = remove_overlapped_pixels(warped_cover, warped_bottom)

    I_ref_warp = warped_bottom + warped_cover

    return (I_in_align, I_ref_warp, I_in_align_landmarks, I_ref_warped_landmarks)


def face_replacement(I_in, I_ref, I_in_landmarks, I_ref_landmarks, parser, useSeamlessClone):
    """
    replace reference image's face to input image's face

    :param I_in: input image
    :param I_ref: reference image
    :param I_in_landmarks: input image's landmark
    :param I_ref_landmarks: reference image's landmark
    :param parser: face parser
    :param useSeamlessClone: is use seamless cloning?
    :return: image which processed face replacement
    """
    I_in_c_s = find_center_pt(I_in_landmarks)
    I_ref_c_s = find_center_pt(I_ref_landmarks)

    # Move Input Face Align Center
    # move_vector = np.array([I_ref_landmarks[30].x, I_ref_landmarks[30].y]) - np.array([I_in_landmarks[30].x, I_in_landmarks[30].y])
    move_vector = I_ref_c_s - I_in_c_s
    M = [[1, 0, move_vector[0]], [0, 1, move_vector[1]]]
    h, w = I_in.shape[:2]
    M = np.float32(M)

    # I_in_move = cv2.warpAffine(I_in, M, (w, h))
    I_in_move = shift_image(I_in, int(np.round(move_vector[0])), int(np.round(move_vector[1])))

    # I_in_align Face ROI
    I_in_face_roi = find_faceRoi_withParser(parser, I_in_move)
    I_in_face_roi = cv2.cvtColor(I_in_face_roi, cv2.COLOR_RGB2GRAY)

    # I_ref_warp Face  ROI
    I_ref_face_roi = find_faceRoi_withParser(parser, I_ref)
    I_ref_face_roi = cv2.cvtColor(I_ref_face_roi, cv2.COLOR_RGB2GRAY)

    # input 이미지와 reference이미지 모두 face 영역인 부분을 찾음
    is_face_both2d = np.logical_and(I_in_face_roi, I_ref_face_roi)
    face_mask = np.array(np.where(is_face_both2d == True, 255, 0), dtype=np.uint8)

    # 얼굴을 바로 가져다 붙이면 배경과 얼굴의 경계가 sharp해지기 때문에 image erosion 수행
    face_mask = cv2.erode(face_mask, kernel=np.ones((3, 3), np.uint8), iterations=3)
    rows, cols = face_mask.shape[0], face_mask.shape[1]
    is_face_both2d = np.expand_dims(np.where(face_mask == 255, True, False), axis=0).reshape(rows, cols, 1)
    is_face_both = np.concatenate([is_face_both2d, is_face_both2d, is_face_both2d], axis=2)

    face_roi_intersection_mask = np.logical_not(is_face_both)
    face_area = np.ma.masked_array(I_in_move, mask=face_roi_intersection_mask, fill_value=0).filled()

    # 배경영역을 찾음
    bg_area = np.ma.masked_array(I_ref, mask=is_face_both, fill_value=0).filled()

    # 이미지 수정
    correct_image = cv2.add(face_area, bg_area)

    # color correction
    if useSeamlessClone == True:
        target_img = np.where(face_mask == 0, 0, 255)
        obj_rec = find_obj_rectangle(target_img)
        x_start = obj_rec[0][0]
        y_start = obj_rec[0][1]

        obj_w = obj_rec[1]
        obj_h = obj_rec[2]

        face_roi_center = (int((x_start + x_start + obj_w) / 2), int((y_start + y_start + obj_h) / 2))

        correct_image = cv2.seamlessClone(src=correct_image,
                                          dst=I_ref,
                                          mask=face_mask,
                                          p=face_roi_center,
                                          flags=cv2.NORMAL_CLONE)

    return correct_image

